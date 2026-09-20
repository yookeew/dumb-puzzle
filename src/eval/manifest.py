"""Run manifests -- fingerprint a holdout eval so a stale/no-op comparison fails
loudly instead of silently comparing a fresh run to a leftover file from an
earlier one.

Motivated by a real incident (PROGRESS.md S14): a retrain script had been
auto-relocated by this repo's housekeeping hook, `python` errored immediately,
and a trailing shell command masked the failure -- the "results" that got read
back were just a stale report from an unrelated earlier experiment. A manifest
comparison would have refused to proceed instead of silently reusing it.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PIPELINE_FILES = [
    ROOT / "src" / "features" / "build_features.py",
    ROOT / "src" / "features" / "encode.py",
    ROOT / "src" / "models" / "fit.py",
]


def _git_sha() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain", "--", *[str(p) for p in PIPELINE_FILES]],
            cwd=ROOT, capture_output=True, text=True, check=True)
        return bool(r.stdout.strip())
    except Exception:
        return True  # fail safe: assume dirty if we can't tell


def _src_hash() -> str:
    h = hashlib.sha256()
    for p in PIPELINE_FILES:
        h.update(p.read_bytes() if p.exists() else b"<missing>")
    return h.hexdigest()[:16]


def build_manifest(*, name: str, n_rows: int, feature_names: list[str], seed: int,
                   valid_month: str, best_iter, clf_best_iter, wall_s: float,
                   **extra) -> dict:
    return dict(
        name=name,
        git_sha=_git_sha(),
        git_dirty=_git_dirty(),
        src_hash=_src_hash(),
        n_rows=n_rows,
        n_features=len(feature_names),
        feature_hash=hashlib.sha256(",".join(sorted(feature_names)).encode()).hexdigest()[:16],
        seed=seed,
        valid_month=valid_month,
        best_iter=best_iter,
        clf_best_iter=clf_best_iter,
        wall_s=round(wall_s, 1),
        **extra,
    )


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_comparable(path_a: Path, path_b: Path) -> tuple[dict, dict]:
    """Load two run manifests and refuse to compare them if the comparison
    would be meaningless.

    Raises if src_hash AND feature_hash are both identical between the two
    runs -- that means same code, same features, nothing for a delta to be
    attributed to. This is the silent-no-op case: a script that didn't
    actually pick up the intended change (or didn't run at all) still leaves
    behind two reports, but their manifests will match exactly.
    """
    a, b = load_manifest(path_a), load_manifest(path_b)
    if a["src_hash"] == b["src_hash"] and a["feature_hash"] == b["feature_hash"]:
        raise RuntimeError(
            f"{path_a.name} and {path_b.name} have identical src_hash and "
            f"feature_hash -- this looks like a no-op (same pipeline code, "
            f"same feature set), not a real A/B comparison. Refusing to diff. "
            f"src_hash={a['src_hash']} feature_hash={a['feature_hash']}"
        )
    if a["git_dirty"] or b["git_dirty"]:
        print(
            f"NOTE: uncommitted changes present during at least one run "
            f"({path_a.name}: dirty={a['git_dirty']}, {path_b.name}: "
            f"dirty={b['git_dirty']}) -- src_hash/feature_hash (not git_sha) "
            f"are the real fingerprint to trust here."
        )
    return a, b
