import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from models.fit import run

fits, ev = run(engine="lgb", target="direct", submit=False, name="lirf_diag")
ev.write_parquet("cache/lirf_diag_ev.parquet")
print("saved cache/lirf_diag_ev.parquet", ev.height)
