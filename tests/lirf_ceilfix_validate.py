import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from models.fit import run

fits, ev = run(engine="lgb", target="mixed", submit=False, name="lirf_ceilfix_mixed")
ev.write_parquet("cache/lirf_ceilfix_mixed_ev.parquet")
print("saved cache/lirf_ceilfix_mixed_ev.parquet", ev.height)
