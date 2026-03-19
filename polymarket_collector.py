#!/usr/bin/env python3
"""
Polymarket BTC 5-Minute Odds Collector — Built for Arya

Price source: GET /midpoint?token_id=...
This is the EXACT endpoint Polymarket's own UI calls, returning the value
shown as "Buy Up X¢ / Buy Down X¢" on the live market page.

Token order: clobTokenIds[0] = Up, clobTokenIds[1] = Down (by spec)
"""

import requests
import csv
import time
import sys
import json
from typing import Optional, Tuple
from datetime import datetime, timezone
from pathlib import Path

# ─── COLORS ───────────────────────────────────────────────────────────────────
RESET  = "\033[0m";  BOLD   = "\033[1m";  DIM    = "\033[2m"
VIOLET = "\033[38;5;135m";  LAVEND = "\033[38;5;183m";  ORANGE = "\033[38;5;208m"
CYAN   = "\033[38;5;87m";   WHITE  = "\033[97m";         GRAY   = "\033[38;5;245m"
GREEN  = "\033[38;5;82m";   RED    = "\033[38;5;203m";   YELLOW = "\033[38;5;220m"
BG_DARK = "\033[48;5;234m"

def c(color, text): return f"{color}{text}{RESET}"

# ─── CONFIG ───────────────────────────────────────────────────────────────────
GAMMA_API      = "https://gamma-api.polymarket.com"
CLOB_API       = "https://clob.polymarket.com"
BINANCE_API    = "https://api.binance.com"
WINDOW_SECONDS = 300
OUTPUT_DIR     = Path(__file__).parent / "odds_data"
RETRY_SLEEP    = 3

OUTPUT_DIR.mkdir(exist_ok=True)
SESSION_FILE = OUTPUT_DIR / f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"

# Prices in CENTS (0–100). These match exactly what Polymarket shows on the UI.
# Source: /midpoint endpoint — same endpoint Polymarket's live page polls.
CSV_HEADERS = [
    "window_id",           # Unix timestamp of window END
    "window_start_utc",
    "window_end_utc",
    "second_in_window",    # 0 = first second, 299 = last second
    "timestamp",
    "datetime_utc",
    "up_cents",            # ← matches "Buy Up X¢"   on Polymarket UI (/midpoint)
    "down_cents",          # ← matches "Buy Down X¢" on Polymarket UI (/midpoint)
    "btc_price_usd",       # live BTC/USDT from Binance
    "market_slug",
    "up_token_id",
    "down_token_id",
]

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def ts_to_dt(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def get_window_end(now=None):
    if now is None: now = int(time.time())
    return ((now // WINDOW_SECONDS) + 1) * WINDOW_SECONDS

def seconds_into_window(now=None):
    if now is None: now = int(time.time())
    return now - (now // WINDOW_SECONDS) * WINDOW_SECONDS

def to_cents(val):
    if val is None: return None
    try:    return round(float(val) * 100, 2)
    except: return None

# ─── POLYMARKET: FIND ACTIVE MARKET ──────────────────────────────────────────
def fetch_market_tokens(window_end):
    """
    Returns (up_token, down_token, slug) for the LIVE window ending at window_end.

    Strategy: try exact slug for current window, then ±1 window (±300s) as
    fallback for boundary timing. The broken slug_contains fallback is removed —
    it returned completely unrelated markets (NBA/NFL) due to a Gamma API bug.

    Validates: active=True, closed=False, and slug contains the expected timestamp.
    Token order: clobTokenIds[0] = Up, clobTokenIds[1] = Down (by Polymarket spec).
    """
    # Try current window first, then adjacent windows in case of boundary drift
    candidates = [window_end, window_end + WINDOW_SECONDS, window_end - WINDOW_SECONDS]

    for end_ts in candidates:
        # Polymarket slugs use window START time, not end time
        window_start = end_ts - WINDOW_SECONDS
        slug = f"btc-updown-5m-{window_start}"
        try:
            resp = requests.get(f"{GAMMA_API}/events?slug={slug}", timeout=10)
            if resp.status_code != 200:
                continue
            events = resp.json()
            if not isinstance(events, list) or not events:
                continue

            # Must be an exact slug match — never accept a mismatch
            event = next((e for e in events if e.get("slug") == slug), None)
            if event is None:
                continue

            markets = event.get("markets", [])
            if not markets:
                continue

            market    = markets[0]
            token_ids = market.get("clobTokenIds", [])
            outcomes  = market.get("outcomes", [])

            if isinstance(token_ids, str): token_ids = json.loads(token_ids)
            if isinstance(outcomes,   str): outcomes  = json.loads(outcomes)

            if len(token_ids) < 2:
                print(c(YELLOW, f"  [!] Only {len(token_ids)} tokens found for {slug}"))
                continue

            # Reject resolved or closed markets (prices would be 0¢ or 100¢)
            if market.get("closed") or market.get("resolved"):
                print(c(YELLOW, f"  [!] Market {slug} is closed/resolved — skipping"))
                continue

            # Print for transparency
            out_prices = market.get("outcomePrices", [])
            if isinstance(out_prices, str): out_prices = json.loads(out_prices)
            print(c(GRAY, f"  [i] outcomes     : {outcomes}"))
            print(c(GRAY, f"  [i] gamma prices : {[f'{float(p)*100:.1f}¢' for p in out_prices]}"))
            if end_ts != window_end:
                print(c(YELLOW, f"  [!] Using adjacent window {slug} (expected {window_end})"))

            return token_ids[0], token_ids[1], slug

        except Exception as e:
            print(c(YELLOW, f"  [!] Error fetching {slug}: {e}"))

    return None, None, f"btc-updown-5m-{window_end - WINDOW_SECONDS}"

# ─── POLYMARKET: MIDPOINT PRICE (matches UI exactly) ─────────────────────────
def get_midpoint(token_id):
    """
    GET /midpoint?token_id=...
    Returns the true probability midpoint in cents (bid+ask)/2.
    This is the EXACT value Polymarket's UI displays as "Buy Up X¢ / Buy Down X¢".
    Confirmed via Chrome DevTools network inspection on the live market page.

    Returns None if the price is out of valid range (0–100¢), which indicates
    a resolved market (0 or 100) or a market that hasn't started yet.
    """
    try:
        r = requests.get(f"{CLOB_API}/midpoint",
                         params={"token_id": token_id}, timeout=5)
        if r.status_code == 200:
            val = to_cents(r.json().get("mid"))
            # Sanity check: live markets trade between ~1¢ and ~99¢
            # 0 or 100 means resolved; skip to avoid logging garbage data
            if val is not None and 0.5 <= val <= 99.5:
                return val
            elif val is not None:
                print(c(YELLOW, f"  [!] Midpoint {val:.1f}¢ looks resolved — skipping"))
    except Exception:
        pass
    return None

# ─── BITCOIN PRICE ────────────────────────────────────────────────────────────
def get_btc_price():
    """Binance BTCUSDT — free, no key, ~100ms latency."""
    try:
        r = requests.get(f"{BINANCE_API}/api/v3/ticker/price",
                         params={"symbol": "BTCUSDT"}, timeout=4)
        if r.status_code == 200:
            return float(r.json()["price"])
    except Exception:
        pass
    # fallback: CoinGecko
    try:
        r = requests.get("https://api.coingecko.com/api/v3/simple/price",
                         params={"ids": "bitcoin", "vs_currencies": "usd"}, timeout=4)
        if r.status_code == 200:
            return float(r.json()["bitcoin"]["usd"])
    except Exception:
        pass
    return None

# ─── DISPLAY ──────────────────────────────────────────────────────────────────
def print_header():
    ARYA = [
        " █████╗  ██████╗  ██╗   ██╗  █████╗ ",
        "██╔══██╗ ██╔══██╗ ╚██╗ ██╔╝ ██╔══██╗",
        "███████║ ██████╔╝  ╚████╔╝  ███████║",
        "██╔══██║ ██╔══██╗   ╚██╔╝   ██╔══██║",
        "██║  ██║ ██║  ██║    ██║    ██║  ██║",
        "╚═╝  ╚═╝ ╚═╝  ╚═╝    ╚═╝    ╚═╝  ╚═╝",
    ]
    W = max(len(l) for l in ARYA)
    pad = 4
    inner = W + pad * 2
    border_top = "  ╔" + "═" * inner + "╗"
    border_bot = "  ╚" + "═" * inner + "╝"
    empty      = "  ║" + " " * inner + "║"
    print()
    print(c(VIOLET+BOLD, border_top))
    print(c(VIOLET+BOLD, empty))
    for line in ARYA:
        lpad = pad + (W - len(line)) // 2
        rpad = inner - len(line) - lpad
        print(c(VIOLET+BOLD, "  ║") + " " * lpad + c(ORANGE+BOLD, line) + " " * rpad + c(VIOLET+BOLD, "║"))
    print(c(VIOLET+BOLD, empty))
    print(c(VIOLET+BOLD, border_bot))
    print()
    print(c(GRAY,"  File   : ")+c(WHITE, SESSION_FILE.name))
    print(c(GRAY,"  Folder : ")+c(WHITE, str(OUTPUT_DIR)))
    print(c(GRAY,"  Prices : ")+c(WHITE, "midpoint = exact match to 'Buy Up X¢ / Buy Down X¢' on Polymarket"))
    print(c(YELLOW,"  Stop   : Ctrl+C"))
    print()

def print_window_banner(window_end, slug):
    ws = window_end - WINDOW_SECONDS
    print(c(VIOLET+BOLD, "  ┌──────────────────────────────────────────────────────┐"))
    print(c(VIOLET+BOLD, "  │")+c(CYAN+BOLD,  f"  ● LIVE: {slug:<44}")+c(VIOLET+BOLD,"│"))
    print(c(VIOLET+BOLD, "  │")+c(GRAY,        f"  {ts_to_dt(ws)}  →  {ts_to_dt(window_end)[:8]} UTC  ")+c(VIOLET+BOLD,"│"))
    print(c(VIOLET+BOLD, "  └──────────────────────────────────────────────────────┘"))
    print()
    print(c(GRAY,f"  {'Sec':>4}  {'Time':>10}")+c(GREEN+BOLD,f"  {'Buy Up':>9}")+c(RED+BOLD,f"  {'Buy Down':>9}")+c(ORANGE+BOLD,f"  {'BTC Price':>12}"))
    print(c(GRAY,f"  {'─'*4}  {'─'*10}  {'─'*9}  {'─'*9}  {'─'*12}"))

def fmt_c(val, color):
    if val is None: return c(GRAY, f"{'N/A':>9}")
    return c(color+BOLD, f"{val:.1f}¢".rjust(9))

def fmt_btc(val):
    if val is None: return c(GRAY, f"{'N/A':>12}")
    return c(ORANGE, f"${val:,.2f}".rjust(12))

def print_tick(second, dt_str, up_cents, down_cents, btc):
    note = c(YELLOW+BOLD, "  ◄ CLOSING") if second >= 240 else ""
    print(c(GRAY,f"  {second:>4}  {dt_str:>10}")+fmt_c(up_cents,GREEN)+fmt_c(down_cents,RED)+fmt_btc(btc)+note)

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
def main():
    print_header()

    with open(SESSION_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        f.flush()

        current_window_end = None
        up_token = down_token = market_slug = None
        windows_collected = total_ticks = 0

        print(c(CYAN, "  Starting — fetching first window...\n"))

        while True:
            now        = int(time.time())
            window_end = get_window_end(now)

            # ── New window? ────────────────────────────────────────────────────
            if window_end != current_window_end:
                current_window_end = window_end
                up_token = down_token = None

                print(c(ORANGE+BOLD, f"\n  [→] Window ends {ts_to_dt(window_end)}"))

                for attempt in range(5):
                    up_token, down_token, market_slug = fetch_market_tokens(window_end)
                    if up_token and down_token: break
                    print(c(YELLOW, f"  [!] Attempt {attempt+1}/5 — retry in {RETRY_SLEEP}s..."))
                    time.sleep(RETRY_SLEEP)

                if not up_token:
                    print(c(RED, "  [✗] Could not find tokens — skipping window."))
                    time.sleep(10)
                    continue

                windows_collected += 1
                print_window_banner(current_window_end, market_slug)

            # ── Fetch midpoint prices (matches Polymarket UI exactly) ──────────
            second = seconds_into_window(now)
            dt_str = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%H:%M:%S")

            up_cents   = get_midpoint(up_token)    # ← matches UI "Buy Up X¢"
            down_cents = get_midpoint(down_token)   # ← matches UI "Buy Down X¢"
            btc        = get_btc_price()

            # ── Write CSV ──────────────────────────────────────────────────────
            ws = current_window_end - WINDOW_SECONDS
            writer.writerow({
                "window_id":        current_window_end,
                "window_start_utc": ts_to_dt(ws),
                "window_end_utc":   ts_to_dt(current_window_end),
                "second_in_window": second,
                "timestamp":        now,
                "datetime_utc":     ts_to_dt(now),
                "up_cents":         up_cents,
                "down_cents":       down_cents,
                "btc_price_usd":    btc,
                "market_slug":      market_slug,
                "up_token_id":      up_token,
                "down_token_id":    down_token,
            })
            f.flush()
            total_ticks += 1

            print_tick(second, dt_str, up_cents, down_cents, btc)

            if second % 60 == 0 and second > 0:
                print()
                print(c(VIOLET,"  ★ ")+c(WHITE+BOLD,str(windows_collected))+c(GRAY," windows  |  ")+c(WHITE+BOLD,str(total_ticks))+c(GRAY," ticks saved"))
                print()

            # Align to wall clock
            sleep_time = (now + 1) - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n  {c(GREEN+BOLD,'[✓]')} {c(WHITE,'Stopped.')}")
        print(c(GRAY, f"  Data saved to: {OUTPUT_DIR}\n"))
        sys.exit(0)
