import argparse
from pathlib import Path

import pandas as pd

from decisionbench.datasets import read_jsonl

parser = argparse.ArgumentParser(
    description="Optional Parquet export; JSONL remains the source of truth"
)
parser.add_argument("run_dir", type=Path)
args = parser.parse_args()
rows = read_jsonl(args.run_dir / "requests.jsonl")
pd.DataFrame(rows).to_parquet(args.run_dir / "requests.parquet", index=False)
print(args.run_dir / "requests.parquet")
