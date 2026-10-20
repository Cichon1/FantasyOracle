# Fantasy Oracle

NFL fantasy projections (PPR) that keep themselves up to date, with rest-of-season rankings and a trade analyzer for
Jackson's Yahoo leagues.

- **Website:** the dashboard in `docs/` (published by GitHub Pages).
- **Model:** `model/` — the weekly projection model (v5.1), rest-of-season rankings and trade values. Details in
  `model/README.md`.
- **Automatic updates:** `.github/workflows/update.yml`. Every 30 minutes it asks each free data source (nflverse stats,
  play-by-play, injury reports, depth charts, rosters, lines, snap counts, charting, expected points) whether it changed.
  When something new has posted it reruns the model and republishes the site: `automation/watch.py` (the check),
  `automation/pipeline.py` (data -> results -> this week's projections -> rest of season -> `docs/data/`),
  `automation/weather.py` (game-time forecasts for outdoor games, Open-Meteo). Trained models are cached between runs,
  so a mid-week update takes a couple of minutes; the first run after new games retrains (about 5 minutes).
- **Force an update:** Actions tab → "Update projections" → Run workflow.

No Claude usage is involved in the automatic updates. Changes to the model itself are proposed as pull requests with
their test results.
