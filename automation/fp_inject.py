"""Add the FantasyPros pulls (fp_rankings.py) to the expert-rankings file the model reads (model/fpdata/db_fpecr.parquet).

Runs inside the update, after download_data.sh has put DynastyProcess's weekly copy in place. For each of the last 14
days with a pull, the day's last pull of FantasyPros' default consensus is written in the same format and page names
as the DynastyProcess copy (weekly: qb.php, ppr-rb.php ...; rest of season: ros-qb.php, ros-ppr-rb.php ...), replacing
any DynastyProcess rows for the same page and date. The model then picks the latest snapshot before each game exactly
as before - it just finds one from the day before instead of the previous Friday.

Does nothing (and the model runs on the DynastyProcess copy alone) when there is no secret or no saved pull.

  python automation/fp_inject.py model/fpdata
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fp_rankings as FR  # noqa: E402

PAGE_TYPE = {"weekly": "weekly-{p}", "ros": "redraft-{p}"}
ECR_TYPE = {"weekly": "wp", "ros": "rp"}


def main(fp_dir):
    if not os.environ.get("FANTASYPROS_API_KEY") or not os.path.exists(FR.STORE):
        print(json.dumps({"fp_inject": "skipped (no key or no saved pull)"}))
        return
    store = FR.load_enc(FR.STORE, {}) or {}
    rows = []
    for day, snap in sorted(store.get("daily", {}).items()):
        for key, players in snap["default"].items():
            kind, pos = key.split("|")
            page = FR.PAGES[(kind, pos)]
            for pid, name, team, ppos, ave, sd, best, worst, ecr, fname in players:
                if ave is None:
                    continue
                rows.append({"fp_page": page, "page_type": PAGE_TYPE[kind].format(p=pos.lower()), "player": name,
                             "id": str(pid), "pos": ppos or pos, "team": team, "ecr": ave, "sd": sd, "best": best,
                             "worst": worst, "mergename": name, "tm": team, "player_filename": fname,
                             "ecr_type": ECR_TYPE[kind], "scrape_date": day})
    if not rows:
        print(json.dumps({"fp_inject": "no rows"}))
        return
    new = pd.DataFrame(rows)
    path = os.path.join(fp_dir, "db_fpecr.parquet")
    old = pd.read_parquet(path)
    key = set(zip(new["fp_page"], new["scrape_date"]))
    keep = ~pd.Series([k in key for k in zip(old["fp_page"], old["scrape_date"].astype(str))], index=old.index)
    for c in old.columns:
        if c not in new.columns:
            new[c] = None
    new = new[old.columns]
    for c in old.columns:
        if str(old[c].dtype) == "float64":
            new[c] = pd.to_numeric(new[c], errors="coerce").astype("float64")
        else:
            new[c] = new[c].astype(old[c].dtype)
    out = pd.concat([old[keep], new], ignore_index=True)
    out.to_parquet(path, index=False)
    print(json.dumps({"fp_inject": len(new), "days": sorted(store.get("daily", {}))[-3:],
                      "replaced_rows": int((~keep).sum())}))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "model", "fpdata"))
