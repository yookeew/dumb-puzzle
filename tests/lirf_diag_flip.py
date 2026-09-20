import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from models.fit import run

fits, ev = run(engine="lgb", target="flip", submit=False, name="lirf_diag_flip")
ev.write_parquet("cache/lirf_diag_flip_ev.parquet")
print("saved cache/lirf_diag_flip_ev.parquet", ev.height)
