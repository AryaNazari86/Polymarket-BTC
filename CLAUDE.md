# Project: Polymarket BTC 5-Minute Prediction Market Analyzer

## Owner
Arya (nazariarya86@gmail.com)

## Project Goal

Arya is building a data-driven edge for trading on **Polymarket's BTC 5-minute up/down prediction markets**.

The core idea:
1. **Collect** live odds data from Polymarket every second, across many consecutive 5-minute windows
2. **Record** how the odds evolve during the first 1–2 minutes of each window
3. **Analyze** whether early-window patterns (e.g. Up at 35¢ in the first 60 seconds) reliably predict the final outcome (Up wins or Down wins)
4. **Use those patterns** to place smarter bets at the start of future windows

This is purely a **data collection → pattern recognition → trading edge** project. The historical Bitcoin OHLC data (from CoinMarketCap) is used as supplemental context for price-level analysis.

---

## What the Bot Does (`polymarket_collector.py`)

### Purpose
Polls Polymarket's live BTC 5-minute market every second and saves the odds + BTC price to CSV.

### How it works
1. **Window detection** — Computes the Unix timestamp of the current 5-minute window's end (`((now // 300) + 1) * 300`). Each Polymarket window has a slug like `btc-updown-5m-{unix_end}`.
2. **Token discovery** — Queries the Gamma API (`https://gamma-api.polymarket.com`) to find the active market and its two CLOB token IDs. `clobTokenIds[0]` = Up outcome, `clobTokenIds[1]` = Down outcome (always, by Polymarket spec).
3. **Price fetch** — Calls `GET https://clob.polymarket.com/midpoint?token_id=...` for each token. This is the **exact endpoint Polymarket's own UI polls**, returning the value shown as **"Buy Up X¢ / Buy Down X¢"** on the live page. Confirmed via Chrome DevTools network inspection.
4. **BTC price** — Fetches live BTC/USDT from Binance (`/api/v3/ticker/price?symbol=BTCUSDT`).
5. **CSV logging** — Writes one row per second to `odds_data/session_YYYYMMDD_HHMMSS.csv`.

### Critical implementation notes
- **Price source is `/midpoint`**, NOT `/price?side=BUY`. The BUY endpoint returns the ASK price (slightly higher). Midpoint = (bid+ask)/2 = what the UI displays. Using the wrong endpoint causes prices to not match the Polymarket website.
- **Window identification**: The correct live window has a red dot (●) in Polymarket's UI and a countdown timer ≤ 5:00. If prices show 0¢ or 99¢, the window is already resolved (too old). If "Price to beat" is blank, the window hasn't started yet (too early).
- **Python 3.9 compatibility**: Use `Optional[X]` from `typing`, not `X | None` syntax.
- **Token order**: `clobTokenIds[0]` = Up, `clobTokenIds[1]` = Down — this is consistent and reliable.

### CSV columns
| Column | Description |
|---|---|
| `window_id` | Unix timestamp of window END |
| `window_start_utc` | Human-readable window start |
| `window_end_utc` | Human-readable window end |
| `second_in_window` | 0 = first second, 299 = last second |
| `timestamp` | Unix timestamp of this tick |
| `datetime_utc` | Human-readable timestamp |
| `up_cents` | "Buy Up X¢" — midpoint price for Up outcome (0–100) |
| `down_cents` | "Buy Down X¢" — midpoint price for Down outcome (0–100) |
| `btc_price_usd` | Live BTC/USD from Binance at this second |
| `market_slug` | e.g. `btc-updown-5m-1773897900` |
| `up_token_id` | CLOB token ID for Up outcome |
| `down_token_id` | CLOB token ID for Down outcome |

---

## Files in This Directory

| File | Description |
|---|---|
| `polymarket_collector.py` | Main bot — run this to collect live data |
| `odds_data/session_*.csv` | Output: one CSV per collection session |
| `Bitcoin_2025-02-01-2026-03-18_historical_data_coinmarketcap.csv` | Historical daily BTC OHLC data (semicolon-delimited) from CoinMarketCap |
| `CLAUDE.md` | This file — project context for Claude |

---

## APIs Used

| API | Purpose | Auth |
|---|---|---|
| `https://gamma-api.polymarket.com` | Market/event discovery, token IDs | None |
| `https://clob.polymarket.com/midpoint` | Live odds prices (matches UI exactly) | None |
| `https://api.binance.com/api/v3/ticker/price` | Live BTC/USDT price | None |

---

## Trading Strategy (IMPORTANT — read carefully)

Arya is NOT trying to predict whether BTC will go up or down at window resolution.

**The actual strategy is:**
1. Watch the first ~1–2 minutes of a window
2. **Buy** either the Up or Down token at some point during that early period
3. **Sell** that token ~30 seconds before the window closes (around second 270)
4. Profit = sell price minus buy price, in cents

The final outcome (which token "wins") does not matter. Arya exits before resolution. This is pure **intra-window price momentum trading** — buying a token cheap and selling it higher within the same 5-minute window.

**Example:** Buy Up at 38¢ at second 70. Sell Up at 54¢ at second 265. Profit = 16¢ per contract. It doesn't matter whether BTC actually went up at second 300.

---

## Next Phase (Analysis)

Once 2+ hours of data are collected (covering ~24+ windows), build `analyze.py` with the following logic:

### Goal
Find: given what the Up/Down prices and BTC price were doing in the **early part of a window**, which token should you buy, at what second, and sell at what second, to maximize profit?

Both the **optimal buy second** and the **optimal sell second** must be discovered by the data — do not hardcode them.

### Step 1 — Optimal sell time discovery
For each window, scan all possible sell times from second 240 to second 285 (last ~60 seconds before close). For each sell time, compute how much profit you would have made if you had bought at second 60 and sold at that sell time, for both Up and Down tokens. Aggregate across all windows to find which sell second (e.g. second 265) produces the highest average profit. That becomes the **target sell second**.

### Step 2 — Optimal buy time discovery
For each window, scan all possible buy times from second 30 to second 150. For each buy time, compute profit = price at target_sell_second minus price at buy_time, for both Up and Down tokens. Find which buy second maximizes average profit across all windows. That becomes the **target buy second**.

### Step 3 — Early signal correlation
Now that you know the optimal buy_second and sell_second, look at what was happening in seconds 0 through buy_second for each window. Extract these features:
- `up_cents` and `down_cents` at second 0, 10, 20, 30
- Slope of `up_cents` from second 0 to second 30 (rising or falling, and by how much)
- Slope of `btc_price_usd` from second 0 to second 30 (BTC trending up or down)
- Whether `up_cents` crossed above or below 50¢ in the early period
- The spread between `up_cents` and `down_cents` at second 30

For each feature pattern, report:
- How many windows matched that pattern
- What % of those windows resulted in a profitable trade (buying Up vs buying Down)
- Average profit in cents for each direction

### Step 4 — Output a trading rule
Synthesize a simple, actionable rule such as:
*"When up_cents is falling in the first 30 seconds AND BTC is trending down, buy Down at second 75. Sell at second 262. Win rate: 71%, avg profit: +9.4¢"*

The output should be a printed report (no charts needed). Use only pandas and standard Python libraries.

---

## How to Run

```bash
cd "Bitcoin polymarket"
python3 polymarket_collector.py
```

Press `Ctrl+C` to stop. Data auto-saves to `odds_data/` every second.
