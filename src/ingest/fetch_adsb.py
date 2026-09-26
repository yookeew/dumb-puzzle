"""Fetch adsb.lol globe_history days and cut them to the challenge airports.

Self-contained (stdlib + numpy + polars + requests; orjson optional) so it can
run as a single file on Colab. For each UTC day:

  1. Pick replicas. The preferred one comes from PREFERRED_RELEASES.txt in the
     root of adsblol/globe_history_<year> (one line per day, comma-separated
     asset URLs of the split tar). Fallbacks, in order, are the release tags
     v<Y.m.d>-planes-readsb-{prod-0, staging-0, prod-0tmp, staging-0tmp}
     looked up through the GitHub API (set GITHUB_TOKEN to lift the 60/h
     unauthenticated rate limit).
  2. Stream the split tar (.aa, .ab, ...) straight from GitHub into a
     streaming tarfile reader -- nothing is extracted to disk, so no working
     directory can leak one day's trace files into the next.
  3. For every traces/**/trace_full_<hex>.json member (gzip-compressed JSON in
     readsb's trace format), keep points inside any airport box
     (+-0.10 deg lat, +-0.15 deg lon around the aerodrome reference point).
  4. Write <out>/adsb_YYYYMMDD.parquet atomically, and append the replica used
     to <out>/manifest.csv. Days already written are skipped, so a
     disconnected Colab session resumes where it stopped.

Corrupt replicas: a structural tar error or a member that fails to
gunzip/parse aborts that replica and the next one is tried from scratch. If
every replica fails, the first is re-read in tolerant mode, skipping bad
members, and the skip count is recorded in the manifest.

Output schema (the "new" schema read by src/ingest/normalise_adsb.py):
  hex str | reg str | ts f64 epoch s | lat f64 | lon f64 | alt_raw f64 (ft,
  null when on ground or unknown) | is_ground bool | gs f64 kt | track f64 deg |
  src str (readsb position source, e.g. adsb_icao, mlat) | airport str

Data licence: ODbL 1.0 (database), feeder contributions CC0 -- see
DATA_SOURCES.md.

Usage:
  python fetch_adsb.py --days 2025-01-01:2025-01-31 2025-07-01:2025-07-31 --out adsb/
  python fetch_adsb.py --days 2026-01-15 --out adsb/
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import io
import json
import os
import sys
import tarfile
import time
import zlib
from pathlib import Path

import numpy as np
import polars as pl
import requests

try:
    import orjson
    _loads = orjson.loads
except ImportError:  # pragma: no cover
    _loads = json.loads

# Aerodrome reference points (lat, lon). Every Gateway stand in
# data/external/stands.csv lies inside the resulting boxes.
ARP = {
    "EDDF": (50.0333, 8.5706), "EDDM": (48.3538, 11.7861), "EGLL": (51.4700, -0.4543),
    "EHAM": (52.3086, 4.7639), "LEBL": (41.2971, 2.0785), "LEMD": (40.4719, -3.5626),
    "LFPG": (49.0097, 2.5479), "LIRF": (41.8003, 12.2389), "LTFM": (41.2753, 28.7519),
    "LSZH": (47.4647, 8.5492),
}
DLAT, DLON = 0.10, 0.15
REPLICA_ORDER = ["prod-0", "staging-0", "prod-0tmp", "staging-0tmp"]
CHUNK = 1 << 20


class CorruptReplica(Exception):
    pass


# ---------------------------------------------------------------- replicas
def _gh_headers() -> dict:
    tok = os.environ.get("GITHUB_TOKEN")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


_preferred_cache: dict[int, dict[str, list[str]]] = {}


def preferred(day: dt.date) -> list[str] | None:
    if day.year not in _preferred_cache:
        url = (f"https://raw.githubusercontent.com/adsblol/globe_history_{day.year}"
               f"/main/PREFERRED_RELEASES.txt")
        m: dict[str, list[str]] = {}
        r = requests.get(url, timeout=60)
        if r.ok:
            for line in r.text.splitlines():
                urls = [u.strip() for u in line.split(",") if u.strip()]
                if urls:
                    tag = urls[0].split("/releases/download/")[1].split("/")[0]
                    m[tag.split("-planes-")[0]] = urls
        _preferred_cache[day.year] = m
    return _preferred_cache[day.year].get(f"v{day:%Y.%m.%d}")


def tag_assets(day: dt.date, replica: str) -> list[str] | None:
    tag = f"v{day:%Y.%m.%d}-planes-readsb-{replica}"
    r = requests.get(f"https://api.github.com/repos/adsblol/globe_history_{day.year}"
                     f"/releases/tags/{tag}", headers=_gh_headers(), timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    urls = sorted(a["browser_download_url"] for a in r.json().get("assets", [])
                  if ".tar" in a["name"])
    return urls or None


def candidates(day: dt.date) -> list[list[str]]:
    out: list[list[str]] = []
    pref = preferred(day)
    if pref:
        out.append(pref)
    for rep in REPLICA_ORDER:
        if pref and f"-planes-readsb-{rep}/" in pref[0]:
            continue
        urls = tag_assets(day, rep)
        if urls:
            out.append(urls)
    return out


# ---------------------------------------------------------------- streaming
class ChainedHTTP(io.RawIOBase):
    """Read the split-tar parts back to back as one stream."""

    def __init__(self, urls: list[str]):
        self.urls = list(urls)
        self.resp = None
        self.it = None
        self.buf = b""
        self.bytes = 0

    def _next(self) -> bool:
        if self.resp is not None:
            self.resp.close()
        if not self.urls:
            return False
        self.resp = requests.get(self.urls.pop(0), stream=True, timeout=120)
        self.resp.raise_for_status()
        self.it = self.resp.iter_content(CHUNK)
        return True

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        while not self.buf:
            if self.it is None and not self._next():
                return 0
            try:
                self.buf = next(self.it)
            except StopIteration:
                self.it = None
                if not self._next():
                    return 0
        n = min(len(b), len(self.buf))
        b[:n] = self.buf[:n]
        self.buf = self.buf[n:]
        self.bytes += n
        return n

    def close(self) -> None:
        if self.resp is not None:
            self.resp.close()
        super().close()


def _in_boxes(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Index into AIRPORTS for each point, -1 when outside every box."""
    idx = np.full(len(lat), -1, dtype=np.int8)
    for k, (a_lat, a_lon) in enumerate(ARP.values()):
        m = (np.abs(lat - a_lat) <= DLAT) & (np.abs(lon - a_lon) <= DLON)
        idx[m & (idx < 0)] = k
    return idx


AIRPORTS = list(ARP)
COLS = ["hex", "reg", "ts", "lat", "lon", "alt_raw", "is_ground", "gs", "track", "src", "airport"]


def parse_day(urls: list[str], tolerant: bool = False) -> tuple[pl.DataFrame, int, int]:
    """Stream one replica; returns (points in boxes, traces read, bad members skipped)."""
    cols: dict[str, list] = {c: [] for c in COLS}
    n_traces = n_bad = 0
    t0 = time.time()
    raw = ChainedHTTP(urls)
    stream = io.BufferedReader(raw, buffer_size=8 * CHUNK)
    try:
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            for m in tar:
                if not (m.isfile() and "trace_full_" in m.name):
                    continue
                try:
                    blob = tar.extractfile(m).read()
                    if blob[:2] == b"\x1f\x8b":
                        blob = gzip.decompress(blob)
                    d = _loads(blob)
                except (OSError, EOFError, zlib.error, ValueError) as e:
                    if not tolerant:
                        raise CorruptReplica(f"{m.name}: {type(e).__name__}: {e}") from e
                    n_bad += 1
                    continue
                n_traces += 1
                if n_traces % 50000 == 0:
                    print(f"    {n_traces:,} traces, {raw.bytes / 1e9:.2f} GB, "
                          f"{len(cols['ts']):,} pts kept, {time.time() - t0:.0f}s", flush=True)
                tr = d.get("trace") or []
                if not tr:
                    continue
                lat = np.fromiter((p[1] for p in tr), float, len(tr))
                lon = np.fromiter((p[2] for p in tr), float, len(tr))
                ap = _in_boxes(lat, lon)
                keep = np.flatnonzero(ap >= 0)
                if not len(keep):
                    continue
                base = float(d.get("timestamp", 0.0))
                hexid = str(d.get("icao", m.name.rsplit("trace_full_", 1)[1].split(".")[0])).lower()
                reg = d.get("r")
                for i in keep:
                    p = tr[i]
                    alt = p[3] if len(p) > 3 else None
                    ground = alt == "ground"
                    cols["hex"].append(hexid)
                    cols["reg"].append(reg)
                    cols["ts"].append(base + float(p[0]))
                    cols["lat"].append(lat[i])
                    cols["lon"].append(lon[i])
                    cols["alt_raw"].append(None if ground or alt is None else float(alt))
                    cols["is_ground"].append(ground)
                    cols["gs"].append(None if len(p) <= 4 or p[4] is None else float(p[4]))
                    cols["track"].append(None if len(p) <= 5 or p[5] is None else float(p[5]))
                    cols["src"].append(p[9] if len(p) > 9 and isinstance(p[9], str) else None)
                    cols["airport"].append(AIRPORTS[ap[i]])
    except (tarfile.ReadError, zlib.error, EOFError, requests.RequestException) as e:
        raise CorruptReplica(f"stream: {type(e).__name__}: {e}") from e
    finally:
        raw.close()
    df = pl.DataFrame(cols, schema={
        "hex": pl.Utf8, "reg": pl.Utf8, "ts": pl.Float64, "lat": pl.Float64, "lon": pl.Float64,
        "alt_raw": pl.Float64, "is_ground": pl.Boolean, "gs": pl.Float64, "track": pl.Float64,
        "src": pl.Utf8, "airport": pl.Utf8})
    print(f"    done: {n_traces:,} traces, {raw.bytes / 1e9:.2f} GB, {df.height:,} pts, "
          f"{time.time() - t0:.0f}s", flush=True)
    return df, n_traces, n_bad


# ---------------------------------------------------------------- driver
def fetch_day(day: dt.date, out: Path) -> None:
    dest = out / f"adsb_{day:%Y%m%d}.parquet"
    if dest.exists():
        print(f"{day}: exists, skipping")
        return
    cands = candidates(day)
    if not cands:
        print(f"{day}: NO RELEASE FOUND on any replica")
        _manifest(out, day, "", "missing", 0, 0, 0)
        return
    result = None
    for urls in cands:
        tag = urls[0].split("/releases/download/")[1].split("/")[0]
        print(f"{day}: trying {tag} ({len(urls)} parts)", flush=True)
        try:
            df, n_tr, n_bad = parse_day(urls)
            result = (tag, df, n_tr, n_bad, "ok")
            break
        except CorruptReplica as e:
            print(f"    corrupt replica, falling back: {e}", flush=True)
    if result is None:
        urls = cands[0]
        tag = urls[0].split("/releases/download/")[1].split("/")[0]
        print(f"{day}: all replicas failed strictly; tolerant re-read of {tag}", flush=True)
        try:
            df, n_tr, n_bad = parse_day(urls, tolerant=True)
            result = (tag, df, n_tr, n_bad, "tolerant")
        except CorruptReplica as e:
            print(f"    tolerant re-read failed too: {e}")
            _manifest(out, day, tag, "failed", 0, 0, 0)
            return
    tag, df, n_tr, n_bad, status = result
    tmp = dest.with_suffix(".parquet.tmp")
    df.write_parquet(tmp)
    tmp.replace(dest)
    _manifest(out, day, tag, status, n_tr, n_bad, df.height)
    print(f"{day}: wrote {dest.name} ({df.height:,} pts, replica {tag}, {status})", flush=True)


def _manifest(out: Path, day, tag, status, n_tr, n_bad, n_pts) -> None:
    path = out / "manifest.csv"
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["day", "release_tag", "status", "traces", "bad_members", "points",
                        "fetched_utc"])
        w.writerow([day.isoformat(), tag, status, n_tr, n_bad, n_pts,
                    dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")])


def parse_days(specs: list[str]) -> list[dt.date]:
    days: list[dt.date] = []
    for s in specs:
        a, _, b = s.partition(":")
        d0 = dt.date.fromisoformat(a)
        d1 = dt.date.fromisoformat(b) if b else d0
        days += [d0 + dt.timedelta(n) for n in range((d1 - d0).days + 1)]
    return days


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", nargs="+", required=True,
                    help="YYYY-MM-DD or YYYY-MM-DD:YYYY-MM-DD (inclusive), several allowed")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "external-data" / "adsb"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for day in parse_days(args.days):
        try:
            fetch_day(day, out)
        except Exception as e:  # keep going: one bad day must not kill a long Colab run
            print(f"{day}: ERROR {type(e).__name__}: {e}", flush=True)
            _manifest(out, day, "", f"error:{type(e).__name__}", 0, 0, 0)


if __name__ == "__main__":
    sys.exit(main())
