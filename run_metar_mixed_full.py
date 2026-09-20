import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from models.fit import run

run(engine="lgb", name="smart-jigsaw_v14", target="mixed", submit=True)
