# AI Context — Polymarket BTC 5-Minute Analyzer

**Owner:** Arya (nazariarya86@gmail.com)

## What This Project Is

A data collection and analysis tool for trading **Polymarket's BTC 5-minute up/down prediction markets**.

The goal: collect live odds every second, find patterns in the first 1–2 minutes of each window, and use those patterns to place smarter bets.

## How It Works

1. Every second, poll the live BTC 5-min Polymarket window
2. Save odds (Up¢ / Down¢) + BTC spot price to CSV
3. After collecting ~24+ windows (2+ hours), analyze early-window patterns vs final outcomes

## Project Structure

| File | Purpose |
|---|---|
| `polymarket_collector.py` | Main bot — run this |
| `odds_data/session_*.csv` | Output CSVs (one per run session) |
| `Bitcoin_2025-02-01-2026-03-18_historical_data_coinmarketcap.csv` | Historical BTC OHLC (semicolon-delimited) |
| `CLAUDE.md` | Full technical spec (primary AI instructions) |

## APIs

| API | Endpoint | Purpose |
|---|---|---|
| Polymarket Gamma | `https://gamma-api.polymarket.com` | Market discovery, token IDs |
| Polymarket CLOB | `https://clob.polymarket.com/midpoint` | Live odds (matches UI exactly) |
| Binance | `https://api.binance.com/api/v3/ticker/price` | Live BTC/USDT price |

No API keys required for any of them.

## Critical Implementation Details

- **Slug format:** `btc-updown-5m-{unix_window_start}` — uses **start** time, not end time
- **Price endpoint:** `/midpoint` (not `/price?side=BUY`) — midpoint matches the Polymarket UI exactly
- **Token order:** `clobTokenIds[0]` = Up, `clobTokenIds[1]` = Down (always)
- **Fallback:** The Gamma API's `slug_contains` filter is broken (returns unrelated markets). Only use exact slug lookups.
- **Python 3.9+** — uses `Optional[X]` from `typing`, not `X | None`

## How to Run

```bash
cd "Bitcoin polymarket"
python3 polymarket_collector.py
# Ctrl+C to stop — data auto-saves to odds_data/
```

## Next Phase (Not Built Yet)

Analysis script to correlate early-window odds patterns with final Up/Down outcomes, using historical BTC OHLC as price-level context.

## Read CLAUDE.md for the Full Spec

`CLAUDE.md` contains the complete technical documentation and is the authoritative reference for this project.
