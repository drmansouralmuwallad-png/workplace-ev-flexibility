"""Rebuild data/raw/*.jsonl and data/processed/*.pkl from the SustainGym extracts in data/sustaingym.

Run this once after cloning. The pickles are written with your installed pandas version.
Filter-flow tables are written to results/<site>_filter_flow[_2021].csv.
"""
import glob, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acndc.sustaingym_source import to_raw_jsonl
from acndc.clean import load_raw, clean

os.makedirs("data/raw", exist_ok=True); os.makedirs("data/processed", exist_ok=True)
cfg = json.load(open("config.json"))

# Check the extracts against the recorded checksums (works on Windows, macOS and Linux)
for line in open("data/sustaingym/SHA256SUMS"):
    digest, path = line.split(maxsplit=1); path = path.strip().lstrip("*")
    got = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if got != digest:
        sys.exit(f"Checksum mismatch for {path}")
print("SustainGym extracts: checksums OK")

for site in ["jpl", "caltech"]:
    for year, suffix in [("2019", ""), ("2021", "_2021")]:
        paths = sorted(glob.glob(f"data/sustaingym/{site}_{year}-*.csv.gz"))
        n = to_raw_jsonl(paths, site, f"data/raw/{site}_{year}.jsonl")
        df, flow = clean(load_raw([f"data/raw/{site}_{year}.jsonl"]), cfg)
        df.to_pickle(f"data/processed/{site}_sessions{suffix}.pkl")
        flow.to_csv(f"results/{site}_filter_flow{suffix}.csv", index=False)
        print(f"{site} {year}: {n} raw sessions -> {len(df)} after cleaning")
