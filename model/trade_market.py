"""FantasyPros Trade Market Values (redraft PPR, from real fantasy trades, last 10 days) for the trade analyzer.
FantasyPros can't be reached from the cloud runs, so the pull happens in Chrome: run fp_trade_market.js on a
www.fantasypros.com page, read the CSV text with get_page_text and save it as fp_market/trade_market_<date>.csv.
`python trade_market.py fp_market/trade_market_<date>.csv dash_out` checks it and writes dash_out/trade_market_v5.csv."""
import sys
import pandas as pd

COLS = ["as_of", "window_start", "fp_id", "player", "pos", "team", "market_value", "market_rank_pos", "trades_10d",
        "value_change_10d", "values_by_day", "trades_by_day", "name_key"]


def check(path, out_dir):
    d = pd.read_csv(path, dtype={"values_by_day": str, "trades_by_day": str})
    missing = [c for c in COLS if c not in d]
    if missing:
        raise SystemExit(f"missing columns: {missing}")
    d = d[d["pos"].isin(["QB", "RB", "WR", "TE"]) & (d["market_value"] > 0)]
    if len(d) < 100:
        raise SystemExit(f"only {len(d)} players - pull looks incomplete")
    d[COLS].to_csv(f"{out_dir}/trade_market_v5.csv", index=False)
    print(len(d), "players as of", d["as_of"].iloc[0], d.groupby("pos").size().to_dict())


if __name__ == "__main__":
    check(sys.argv[1], sys.argv[2])
