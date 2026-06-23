# Touchline

A football prediction system combining a **pre-match score/winner model**
(Dixon-Coles) and a **live in-play win-probability model**, served through
a real-time web app with a chat interface for natural-language queries.

The name nods to the live, on-the-sideline angle of the project — the
in-play model updates as the match unfolds, the way a touchline view does.

## Why this exists

Most portfolio prediction projects stop at "trained a classifier in a
notebook." This one trains two genuinely different models, serves them
through a live API, and visualizes predictions as a match unfolds —
closer to how this is actually done in sports analytics.

## Architecture

```
StatsBomb open data ──► feature engineering ──► Dixon-Coles (pre-match)
                                              └► classifier (in-play)
                                                        │
API-Football (live) ──► polling/cache layer ──► inference service ──► WebSocket
                                                                          │
                                                                  React frontend
                                                          (chart / pitch / scrubber)
                                                                          │
                                                              chat interface (router)
```

## Project status

See [`docs/MASTERPLAN.md`](docs/MASTERPLAN.md) for the week-by-week build
checklist and current progress.

## Stack

- **Data:** StatsBomb open data (`statsbombpy`), API-Football (live, free tier)
- **Models:** Dixon-Coles (pre-match), logistic regression/XGBoost (in-play)
- **Backend:** FastAPI / ASP.NET Core + WebSocket hub
- **Frontend:** React, D3, animated SVG pitch visualization
- **Hosting:** Vercel/Render (frontend), Render free tier (backend + Postgres)

## Local setup

_To be filled in as the backend/frontend take shape._

## Methodology notes

Model validation, calibration plots, and feature rationale are documented
in [`docs/`](docs/) as each piece is built.
