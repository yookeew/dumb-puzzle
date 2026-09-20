import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from models.fit import run

fits, ev = run(engine="lgb", target="flip", loss="quantile", huber_alpha=0.5,
                submit=False, name="lirf_quantile_median")
ev.write_parquet("cache/lirf_quantile_ev.parquet")
print("saved cache/lirf_quantile_ev.parquet", ev.height)
