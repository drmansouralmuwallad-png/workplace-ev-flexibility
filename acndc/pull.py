"""Pull raw ACN-Data sessions via the public API (free registration token required).

Raw documents are stored unparsed, one JSON object per line, together with a
metadata file recording the pull timestamp, query and session counts so the
extraction can be reported exactly in the Data section.
"""
import json
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests


def _http_date(ts_utc: pd.Timestamp) -> str:
    return ts_utc.strftime("%a, %d %b %Y %H:%M:%S GMT")


def pull_site(site, start_local, end_local, cfg, out_dir, tz="America/Los_Angeles"):
    token = os.environ.get(cfg["api"]["token_env_var"])
    if not token:
        raise RuntimeError(
            f"Set the {cfg['api']['token_env_var']} environment variable to your ACN-Data API token "
            "(register at https://ev.caltech.edu/dataset).")
    start = pd.Timestamp(start_local, tz=tz).tz_convert("UTC")
    end = pd.Timestamp(end_local, tz=tz).tz_convert("UTC")
    cond = f'connectionTime >= "{_http_date(start)}" and connectionTime < "{_http_date(end)}"'
    base = cfg["api"]["base_url"]
    url = f"{base}sessions/{site}?where={cond}&sort=connectionTime&max_results=100"

    os.makedirs(out_dir, exist_ok=True)
    tag = f"{site}_{start_local}_{end_local}"
    raw_path = os.path.join(out_dir, f"{tag}.jsonl")
    n = 0
    with open(raw_path, "w") as f:
        while url:
            for attempt in range(5):
                r = requests.get(url, auth=(token, ""), timeout=60)
                if r.status_code == 200:
                    break
                time.sleep(2 ** attempt)
            r.raise_for_status()
            payload = r.json()
            for s in payload.get("_items", []):
                f.write(json.dumps(s) + "\n")
                n += 1
            nxt = payload.get("_links", {}).get("next")
            url = base + nxt["href"] if nxt else None
    meta = {
        "site": site, "start_local": start_local, "end_local": end_local, "where": cond,
        "n_sessions_raw": n, "pulled_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_base": base,
    }
    with open(os.path.join(out_dir, f"{tag}.meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return raw_path, meta
