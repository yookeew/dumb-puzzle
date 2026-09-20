import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from models.fit import run

fits, ev = run(engine="lgb", target="direct", submit=False, name="echoclf_hour_test")
ev.write_parquet("cache/echoclf_hour_test_ev.parquet")
print("saved cache/echoclf_hour_test_ev.parquet", ev.height)
