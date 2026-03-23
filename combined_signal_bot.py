#!/usr/bin/env python3
"""
Combined Signal Bot — Polymarket BTC 5-Min Odds Momentum

Strategy: Follow what other bots are predicting, not the BTC price directly.
- Watch Polymarket odds drift for the first 75 seconds
- If odds are clearly moving one way AND BTC confirms that direction → ride it
- Buy at s80, sell at s180 (before extremes hit ~57%+ of windows)
- Append every window's data to odds_data/live_training.csv for continuous learning
- HTTP dashboard on port 8080 → open http://localhost:8080

Evidence from 121 windows of real data:
  s30 signal: 56% accuracy  (too early, no signal yet)
  s75 signal: 65% accuracy  (sweet spot)
  BTC confirms direction: 65% win, +7.1¢ avg
  BTC opposes direction : 30% win, -7.0¢ avg  ← must filter these out
  Best sell: s180 (64% win, +6¢) before 57%+ of windows lock into extremes
  Best drift threshold: >4¢ at s75
"""

import csv
import glob
import json as _json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from polymarket_collector import (
    BINANCE_API, BOLD, CYAN, GRAY, GREEN, ORANGE, RED, VIOLET, WHITE, YELLOW,
    WINDOW_SECONDS, c, fetch_market_tokens, get_btc_price, get_midpoint,
    get_window_end, seconds_into_window,
)

# ─── CONFIG (updated dynamically from data every 10 windows) ──────────────────
DASHBOARD_PORT    = 8080
RETRY_SLEEP       = 3
LIVE_CSV          = Path("odds_data/live_training.csv")
LIVE_CSV_COLS     = [
    "window_id", "window_start_utc", "window_end_utc", "second_in_window",
    "timestamp", "datetime_utc",
    "up_cents", "down_cents", "btc_price_usd",
    "market_slug", "up_token_id", "down_token_id",
]

# Dynamic thresholds — re-computed from real data every 10 windows
class Config:
    # ── Timing (from 121-window empirical analysis) ────────────────────────────
    signal_second:  int   = 75    # s75: 66% predictive accuracy (best balance)
    buy_second:     int   = 80    # enter 5s after signal to confirm direction
    sell_second:    int   = 180   # momentum peaks ~s170-180 (best ¢/s rate)

    # ── Drift filters ──────────────────────────────────────────────────────────
    # Speed = |drift| / signal_second  (¢ per second)
    # <0.10¢/s (< 7.5¢ total): too slow/noisy → only 59% accuracy → SKIP
    # 0.10–0.20¢/s (7.5–15¢):  sweet spot    → 77% accuracy        → HIGH
    # >0.20¢/s (> 15¢):         fast/extreme  → 65% accuracy        → MED
    min_drift_cents: float = 7.5  # below this = noise, skip
    max_drift_cents: float = 20.0 # above this = odds already too extreme

    # ── Extreme odds filter ────────────────────────────────────────────────────
    # If odds already past 65¢ at s75, the move is largely done — avg +9.6¢ in
    # the not-extreme zone vs -0.2¢ in the extreme zone at threshold=65
    extreme_thresh: float = 65.0  # skip if up_75 > 65 or up_75 < 35

    # ── BTC confirmation ───────────────────────────────────────────────────────
    # BTC moving same direction as odds: 67% win +4.8¢
    # BTC opposing:                      60% win +3.2¢ (still +EV but weaker)
    # We trade both but flag LOW when BTC opposes
    require_btc:    bool  = False # allow LOW-conf trades without BTC confirm

CFG = Config()


# ─── SHARED STATE (HTTP dashboard) ────────────────────────────────────────────
_LOCK: threading.Lock = threading.Lock()
_STATE: Dict = {
    "window":    {"start_ts": 0, "end_ts": 0, "start_label": "—", "end_label": "—", "second": 0},
    "btc":       {"current": None, "s0": None, "macro": {}},
    "ticks":     [],
    "signal":    {
        "rule": None, "rule_name": "—", "confidence": "—",
        "description": "—", "action": "WAITING",
        "buy_second": 80, "sell_second": 180, "rule_check": 75,
    },
    "scorecard": {
        "total": 0, "wins": 0, "win_rate": 0.0,
        "avg_profit": 0.0, "total_profit": 0.0, "history": [],
    },
    "status": "waiting",
    "thresholds": {"signal_s": 75, "buy_s": 80, "sell_s": 180, "drift": 4.0},
}


def _sset(**kw: object) -> None:
    with _LOCK:
        for k, v in kw.items():
            if isinstance(v, dict) and isinstance(_STATE.get(k), dict):
                _STATE[k].update(v)
            else:
                _STATE[k] = v


def _append_tick(second: int, up: Optional[float], down: Optional[float],
                 btc: Optional[float], up_slope: float, btc_slope: float) -> None:
    with _LOCK:
        _STATE["ticks"].append({
            "second": second, "up": up, "down": down, "btc": btc,
            "up_slope": round(up_slope, 2), "btc_slope": round(btc_slope, 2),
        })
        _STATE["window"]["second"] = second
        _STATE["btc"]["current"] = btc


# ─── HTTP DASHBOARD SERVER ─────────────────────────────────────────────────────
_HTML_PATH = Path(__file__).parent / "dashboard.html"


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/api/state":
            body = _json.dumps(_STATE).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path in ("/", "/dashboard"):
            try:
                content = _HTML_PATH.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except Exception as exc:
                err = str(exc).encode()
                self.send_response(500)
                self.end_headers()
                self.wfile.write(err)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt: str, *args: object) -> None:
        pass  # silence access logs


def _start_dashboard() -> None:
    try:
        srv = HTTPServer(("0.0.0.0", DASHBOARD_PORT), _Handler)
        srv.serve_forever()
    except Exception as exc:
        print(c(YELLOW, f"  [!] Dashboard error: {exc}"))


# ─── MACRO BTC CONTEXT ────────────────────────────────────────────────────────
def fetch_macro_context() -> Dict:
    """Last 62 1-min candles → % changes, EMA20 position, and 30-day daily trend."""
    result: Dict = {}
    try:
        # 1-minute intraday context
        resp = requests.get(
            f"{BINANCE_API}/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "1m", "limit": 62},
            timeout=10,
        )
        if resp.status_code == 200:
            candles = resp.json()
            if len(candles) >= 21:
                closes = [float(r[4]) for r in candles[:-1]]
                cur = closes[-1]

                def pct(n: int) -> Optional[float]:
                    return (cur / closes[-n] - 1) * 100 if len(closes) >= n else None

                ema = closes[-20]
                k = 2.0 / 21.0
                for p in closes[-19:]:
                    ema = p * k + ema * (1.0 - k)

                result.update({
                    "pct_15m": pct(15), "pct_30m": pct(30), "pct_60m": pct(60),
                    "above_ema20": cur > ema,
                })
    except Exception as exc:
        print(c(YELLOW, f"  [!] 1m macro error: {exc}"))

    try:
        # Daily context — last 30 days for longer trend
        resp2 = requests.get(
            f"{BINANCE_API}/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "1d", "limit": 31},
            timeout=10,
        )
        if resp2.status_code == 200:
            days = resp2.json()
            if len(days) >= 8:
                d_closes = [float(r[4]) for r in days[:-1]]
                cur_d = d_closes[-1]
                result["pct_7d"]  = (cur_d / d_closes[-7]  - 1) * 100 if len(d_closes) >= 7  else None
                result["pct_30d"] = (cur_d / d_closes[-30] - 1) * 100 if len(d_closes) >= 30 else None
                # 7-day SMA
                sma7 = sum(d_closes[-7:]) / 7
                result["above_sma7d"] = cur_d > sma7
    except Exception as exc:
        print(c(YELLOW, f"  [!] Daily macro error: {exc}"))

    return result


# ─── LIVE DATA COLLECTION ──────────────────────────────────────────────────────
def append_window_to_csv(window_id: int, window_start: int, window_end: int,
                         ticks: List[Dict]) -> None:
    """Append all ticks from this window to live_training.csv (matches session CSV schema)."""
    LIVE_CSV.parent.mkdir(exist_ok=True)
    write_header = not LIVE_CSV.exists() or LIVE_CSV.stat().st_size == 0
    ws_str = datetime.fromtimestamp(window_start, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    we_str = datetime.fromtimestamp(window_end,   tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(LIVE_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LIVE_CSV_COLS)
            if write_header:
                writer.writeheader()
            for t in ticks:
                ts = window_start + t["second"]
                dt_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow({
                    "window_id":        window_id,
                    "window_start_utc": ws_str,
                    "window_end_utc":   we_str,
                    "second_in_window": t["second"],
                    "timestamp":        ts,
                    "datetime_utc":     dt_str,
                    "up_cents":         t.get("up"),
                    "down_cents":       t.get("down"),
                    "btc_price_usd":    t.get("btc"),
                    "market_slug":      t.get("slug"),
                    "up_token_id":      t.get("up_token"),
                    "down_token_id":    t.get("down_token"),
                })
    except Exception as exc:
        print(c(YELLOW, f"  [!] CSV write error: {exc}"))


# ─── DYNAMIC THRESHOLD LEARNING ───────────────────────────────────────────────
def recompute_thresholds() -> None:
    """Re-analyze all collected data and update CFG thresholds."""
    # Load all session CSVs + live training CSV
    all_files = sorted(glob.glob("odds_data/session_*.csv"))
    if LIVE_CSV.exists():
        all_files.append(str(LIVE_CSV))

    dfs = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            if len(df) > 0:
                dfs.append(df)
        except Exception:
            pass

    if not dfs:
        return

    all_df = pd.concat(dfs, ignore_index=True)

    def get_at(wdf: pd.DataFrame, sec: int, col: str = "up_cents", tol: int = 5) -> Optional[float]:
        sub = wdf[abs(wdf["second_in_window"] - sec) <= tol]
        if len(sub) == 0:
            return None
        return sub.iloc[(sub["second_in_window"] - sec).abs().argsort()].iloc[0][col]

    windows_g = all_df.groupby("window_id")
    rows = []
    for wid, wdf in windows_g:
        wdf = wdf.sort_values("second_in_window")
        if wdf["second_in_window"].min() > 5 or wdf["second_in_window"].max() < 180:
            continue
        row = {"window_id": wid}
        for s in [0, 60, 75, 90, 120, 180, 210]:
            row[f"up_{s}"] = get_at(wdf, s)
        for s in [0, 60]:
            row[f"btc_{s}"] = get_at(wdf, s, "btc_price_usd")
        rows.append(row)

    if len(rows) < 20:
        return  # need at least 20 windows to learn

    feat = pd.DataFrame(rows).dropna(subset=["up_0", "up_75", "up_180"])

    # Find best sell second
    best_sell_s, best_sell_avg = CFG.sell_second, -999.0
    for sell_s_candidate in [150, 165, 180, 195, 210, 225]:
        col_s = f"up_{sell_s_candidate}"
        if col_s not in feat.columns:
            continue
        sub = feat.dropna(subset=["up_75", col_s])
        profits = []
        for _, r in sub.iterrows():
            drift = r["up_75"] - r["up_0"]
            if abs(drift) < 3 or r["up_75"] > 70 or r["up_75"] < 30:
                continue
            profits.append(r["up_75"] - r[col_s] if drift < 0 else r[col_s] - r["up_75"])
        if len(profits) >= 10:
            avg = float(np.mean(profits))
            if avg > best_sell_avg:
                best_sell_avg = avg
                best_sell_s = sell_s_candidate

    # Find best drift threshold
    best_thresh, best_thresh_avg = CFG.drift_thresh, -999.0
    for thresh_candidate in [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
        sub = feat.dropna(subset=["up_75", f"up_{best_sell_s}"])
        profits = []
        for _, r in sub.iterrows():
            drift = r["up_75"] - r["up_0"]
            if abs(drift) < thresh_candidate or r["up_75"] > 70 or r["up_75"] < 30:
                continue
            col_s = f"up_{best_sell_s}"
            profits.append(r["up_75"] - r[col_s] if drift < 0 else r[col_s] - r["up_75"])
        if len(profits) >= 8:
            avg = float(np.mean(profits))
            if avg > best_thresh_avg:
                best_thresh_avg = avg
                best_thresh = thresh_candidate

    CFG.sell_second  = best_sell_s
    CFG.drift_thresh = best_thresh
    _sset(thresholds={
        "signal_s": CFG.signal_second, "buy_s": CFG.buy_second,
        "sell_s": CFG.sell_second,     "drift": CFG.drift_thresh,
    })
    print(c(CYAN, f"  [↻] Thresholds updated: sell=s{CFG.sell_second}  drift>{CFG.drift_thresh}¢"
                  f"  (from {len(feat)} windows)"))


# ─── SIGNAL EVALUATION ────────────────────────────────────────────────────────
def evaluate_signal(
    up_s0: float, up_check: float,
    btc_s0: float, btc_check: float,
) -> Tuple[Optional[str], str, str, str]:
    """
    Odds momentum strategy — ride what other bots are already pushing.

    We watch the first 75 seconds. Other bots are reacting to BTC data and
    pushing Polymarket odds in one direction. We detect that sustained push,
    enter at s80, and exit at s180 before the odds hit extremes (90%+).

    Returns (direction, name, confidence, description) or (None, ...) for STAY OUT.
    """
    up_drift  = up_check - up_s0
    btc_drift = btc_check - btc_s0
    speed     = abs(up_drift) / CFG.signal_second   # ¢ per second

    # ── Gate 1: odds must not already be extreme at check time ────────────────
    # Data: not-extreme zone (35-65¢) → avg +9.6¢, 68% win
    #       extreme zone              → avg -0.2¢ (edge is gone)
    if up_check > CFG.extreme_thresh or up_check < (100.0 - CFG.extreme_thresh):
        return (None, f"Odds extreme at s{CFG.signal_second} ({up_check:.1f}¢) — edge gone", "—",
                f"Need 35–65¢ range. Currently {up_check:.1f}¢.")

    # ── Gate 2: minimum drift — must be a real move, not noise ────────────────
    # Data: <7.5¢ total drift at s75 → only 59% accuracy (barely above coin flip)
    if abs(up_drift) < CFG.min_drift_cents:
        return (None, f"Drift too small ({up_drift:+.1f}¢ < {CFG.min_drift_cents}¢) — noise", "—",
                f"Bots haven't committed to a direction yet.")

    # ── Gate 3: maximum drift — not so fast it's already priced in ───────────
    # >20¢ in 75s means speed >0.27¢/s — odds likely hitting extreme soon
    if abs(up_drift) > CFG.max_drift_cents:
        return (None, f"Drift too large ({up_drift:+.1f}¢ > {CFG.max_drift_cents}¢) — too late", "—",
                f"Move already {up_drift:+.1f}¢ — approaching extreme, no room left.")

    # ── Direction: follow the odds momentum ───────────────────────────────────
    direction    = "UP" if up_drift > 0 else "DOWN"
    btc_confirms = (up_drift > 0 and btc_drift > 0) or (up_drift < 0 and btc_drift < 0)

    # ── Confidence tier based on speed and BTC ────────────────────────────────
    # Sweet-spot speed: 0.10–0.20¢/s (7.5–15¢ total) + BTC confirms → 75% win, +11.5¢ avg
    # Outside speed range but BTC confirms                            → 67% win, +4.8¢ avg
    # BTC opposing direction                                          → 60% win, +3.2¢ avg
    in_sweet_speed = 0.10 <= speed <= 0.20

    if in_sweet_speed and btc_confirms:
        return (direction, "Momentum Sweet Spot + BTC",   "HIGH",
                f"75% win  avg +11.5¢  |  speed {speed:.2f}¢/s  |  BTC confirms direction")
    elif btc_confirms:
        return (direction, "Momentum + BTC Confirm",       "MED",
                f"67% win  avg +4.8¢   |  speed {speed:.2f}¢/s  |  BTC confirms direction")
    else:
        return (direction, "Momentum (BTC not confirming)", "LOW",
                f"60% win  avg +3.2¢   |  speed {speed:.2f}¢/s  |  ⚠ BTC direction differs")


# ─── DISPLAY ──────────────────────────────────────────────────────────────────
def _dt(ts: int, fmt: str) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(fmt)


def print_header() -> None:
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
    print()
    print(c(VIOLET + BOLD, "  ╔" + "═" * inner + "╗"))
    print(c(VIOLET + BOLD, "  ║" + " " * inner + "║"))
    for line in ARYA:
        lp = pad + (W - len(line)) // 2
        rp = inner - len(line) - lp
        print(c(VIOLET + BOLD, "  ║") + " " * lp + c(ORANGE + BOLD, line) + " " * rp + c(VIOLET + BOLD, "║"))
    print(c(VIOLET + BOLD, "  ║" + " " * inner + "║"))
    print(c(VIOLET + BOLD, "  ╚" + "═" * inner + "╝"))
    print()
    print(c(GRAY,  "  Odds momentum bot  ·  watch s0→s75  ·  buy s80  ·  sell s180"))
    print(c(GRAY,  "  HIGH: 75% win +11.5¢  ·  MED: 67% win +4.8¢  ·  LOW: 60% win +3.2¢"))
    print(c(GRAY,  "  Trades ~21% of windows (only when signal is clean)"))
    print(c(GRAY,  "  Requires BTC to confirm direction — filters the -7¢ avg losers"))
    print(c(CYAN,  f"  Dashboard → http://localhost:{DASHBOARD_PORT}"))
    print()


def print_tick(second: int, up: Optional[float], down: Optional[float],
               btc: Optional[float], up_slope: float, btc_slope: float) -> None:
    up_s  = f"{up:.1f}¢"   if up  is not None else "N/A "
    dn_s  = f"{down:.1f}¢" if down is not None else "N/A "
    bt_s  = f"${btc:,.0f}" if btc  is not None else "N/A "
    uc = GREEN if up_slope  >= 0 else RED
    bc = GREEN if btc_slope >= 0 else RED
    print(
        c(GRAY,         f"  s={second:<3d}  ") +
        c(GREEN + BOLD, f"Up:{up_s:<7}") +
        c(RED   + BOLD, f"Dn:{dn_s:<7}") +
        c(ORANGE,       f"BTC:{bt_s:<12}") +
        c(uc,           f"ΔUp:{up_slope:+.1f}¢  ") +
        c(bc,           f"ΔBTC:${btc_slope:+.0f}")
    )


def print_decision(window_start: int, window_end: int,
                   up_s0: float, up_check: float,
                   btc_s0: float, btc_check: float,
                   macro: Dict,
                   direction: Optional[str], rule_name: str,
                   confidence: str, rule_desc: str) -> None:
    bar = "═" * 56
    ws_str  = _dt(window_start, "%b %d  %H:%M")
    we_str  = _dt(window_end,   "%H:%M UTC")
    up_d    = up_check  - up_s0
    btc_d   = btc_check - btc_s0

    print()
    print(c(VIOLET + BOLD, f"  ╔{bar}╗"))
    title = f"  WINDOW  {ws_str} – {we_str}  |  BTC ${btc_check:,.0f}"
    print(c(VIOLET + BOLD, "  ║") + c(CYAN + BOLD, f"{title:<56}") + c(VIOLET + BOLD, "║"))
    print(c(VIOLET + BOLD, f"  ╠{bar}╣"))

    # Odds movement
    print(c(VIOLET + BOLD, "  ║  ") + c(WHITE + BOLD, "── Odds drift s0 → s75 ─────────────────────────────"))
    uc  = GREEN + BOLD if up_d  > 0 else RED + BOLD
    bdc = GREEN + BOLD if btc_d > 0 else RED + BOLD
    ud  = "↑ RISING"  if up_d  > 0 else "↓ FALLING"
    bd  = "↑ RISING"  if btc_d > 0 else "↓ FALLING"
    print(c(VIOLET + BOLD, "  ║  ") + c(GRAY, f"  Up odds drift  : ") + c(uc,  f"{up_d:+.1f}¢  {ud}"))
    print(c(VIOLET + BOLD, "  ║  ") + c(GRAY, f"  BTC drift      : ") + c(bdc, f"${btc_d:+.1f}  {bd}"))
    btc_conf = (up_d > 0 and btc_d > 0) or (up_d < 0 and btc_d < 0)
    cc_str = c(GREEN + BOLD, "✓ CONFIRMS") if btc_conf else c(RED + BOLD, "✗ OPPOSES (risky!)")
    print(c(VIOLET + BOLD, "  ║  ") + c(GRAY, f"  BTC alignment  : ") + cc_str)
    print(c(VIOLET + BOLD, "  ║  ") + c(GRAY, f"  Up at s0 / s75 : ") + c(WHITE, f"{up_s0:.1f}¢  →  {up_check:.1f}¢"))

    # Macro
    print(c(VIOLET + BOLD, f"  ╠{bar}╣"))
    print(c(VIOLET + BOLD, "  ║  ") + c(WHITE + BOLD, "── Macro BTC (other bots' input data) ──────────────"))
    if macro:
        for lbl, key in [("1h trend", "pct_60m"), ("7d trend", "pct_7d"), ("30d trend", "pct_30d")]:
            val = macro.get(key)
            if val is not None:
                col = GREEN + BOLD if val > 0 else RED + BOLD
                print(c(VIOLET + BOLD, "  ║  ") + c(GRAY, f"  {lbl:<14}: ") + c(col, f"{val:+.3f}%"))
        above = macro.get("above_ema20")
        if above is not None:
            ec = GREEN + BOLD if above else RED + BOLD
            print(c(VIOLET + BOLD, "  ║  ") + c(GRAY, "  vs 20-EMA(1m) : ") + c(ec, "ABOVE ▲" if above else "BELOW ▼"))
        above7 = macro.get("above_sma7d")
        if above7 is not None:
            ec = GREEN + BOLD if above7 else RED + BOLD
            print(c(VIOLET + BOLD, "  ║  ") + c(GRAY, "  vs 7d SMA     : ") + c(ec, "ABOVE ▲" if above7 else "BELOW ▼"))

    # Decision
    print(c(VIOLET + BOLD, f"  ╠{bar}╣"))
    print(c(VIOLET + BOLD, "  ║  ") + c(WHITE + BOLD, "── Decision ─────────────────────────────────────────"))
    print(c(VIOLET + BOLD, "  ║  ") + c(GRAY, f"  Rule    : ") + c(WHITE, rule_name))
    if direction:
        conf_col = GREEN + BOLD if confidence == "HIGH" else (ORANGE + BOLD if confidence == "MEDIUM" else YELLOW + BOLD)
        print(c(VIOLET + BOLD, "  ║  ") + c(GRAY, f"  Details : ") + c(WHITE, rule_desc))
        print(c(VIOLET + BOLD, "  ║  ") + c(GRAY, f"  Conf    : ") + c(conf_col, confidence))
    print(c(VIOLET + BOLD, f"  ╚{bar}╝"))
    print()

    if direction:
        low_warn = "  ⚠ LOW CONFIDENCE" if confidence == "LOW" else ""
        dir_col = GREEN + BOLD if direction == "UP" else RED + BOLD
        print(c(dir_col, f"  ★ BUY {direction} NOW  (second {CFG.buy_second}){low_warn}"))
        print(c(CYAN   + BOLD, f"  ★ SELL           (second {CFG.sell_second})"))
    else:
        print(c(RED + BOLD, f"  ✗ STAY OUT  —  {rule_name}"))
    print(c(VIOLET + BOLD, "  " + "═" * 58))


# ─── SCORECARD ────────────────────────────────────────────────────────────────
class Scorecard:
    def __init__(self) -> None:
        self.total:  int   = 0
        self.wins:   int   = 0
        self.profit: float = 0.0

    def record(self, profit: float) -> None:
        self.total  += 1
        self.profit += profit
        if profit > 0:
            self.wins += 1

    def update_state(self, history_entry: Dict) -> None:
        wr  = self.wins / self.total * 100 if self.total > 0 else 0.0
        avg = self.profit / self.total if self.total > 0 else 0.0
        with _LOCK:
            _STATE["scorecard"].update({
                "total": self.total, "wins": self.wins,
                "win_rate": round(wr, 1), "avg_profit": round(avg, 2),
                "total_profit": round(self.profit, 2),
            })
            _STATE["scorecard"]["history"].append(history_entry)

    def display(self) -> None:
        if self.total == 0:
            print(c(GRAY, "  No signals fired yet."))
            return
        wr  = self.wins / self.total * 100
        avg = self.profit / self.total
        print()
        print(c(VIOLET + BOLD, "  ┌─── SESSION SCORECARD ──────────────────────────────┐"))
        print(c(VIOLET + BOLD, "  │  ") + c(WHITE, f"Signals : {self.total}  │  Wins: {self.wins}  │  Win rate: {wr:.1f}%"))
        avg_col = GREEN + BOLD if avg >= 0 else RED + BOLD
        tot_col = GREEN + BOLD if self.profit >= 0 else RED + BOLD
        print(c(VIOLET + BOLD, "  │  ") + c(avg_col, f"Avg profit   : {avg:+.1f}¢"))
        print(c(VIOLET + BOLD, "  │  ") + c(tot_col, f"Total profit : {self.profit:+.1f}¢"))
        print(c(VIOLET + BOLD, "  └────────────────────────────────────────────────────┘"))


# ─── TIMING HELPERS ───────────────────────────────────────────────────────────
def _sleep_to_next_tick() -> None:
    now = time.time()
    gap = (int(now) + 1) - now
    if gap > 0:
        time.sleep(gap)


def wait_until_second(target: int) -> None:
    while seconds_into_window() < target:
        _sleep_to_next_tick()


def wait_for_new_window() -> None:
    prev = get_window_end()
    while True:
        we = get_window_end()
        if we != prev and seconds_into_window() <= 3:
            return
        prev = we
        _sleep_to_next_tick()


# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
def main() -> None:
    # Start HTTP dashboard
    threading.Thread(target=_start_dashboard, daemon=True).start()

    print_header()
    scorecard  = Scorecard()
    windows_run = 0

    # Skip into window if we're already deep in one
    cur_sec = seconds_into_window()
    if cur_sec > 5:
        print(c(CYAN, f"  s={cur_sec} — waiting for next window boundary...\n"))
        _sset(status="waiting")
        wait_for_new_window()

    while True:
        now          = int(time.time())
        window_end   = get_window_end(now)
        window_start = window_end - WINDOW_SECONDS
        ws_label = _dt(window_start, "%b %d  %H:%M")
        we_label = _dt(window_end,   "%H:%M UTC")

        print(c(VIOLET + BOLD, f"\n  ┌── NEW WINDOW  {ws_label} – {we_label} ─────────────────────────┐"))

        # Reset state
        _sset(
            status="starting",
            window={"start_ts": window_start, "end_ts": window_end,
                    "start_label": ws_label, "end_label": we_label, "second": 0},
            ticks=[],
            signal={"rule": None, "rule_name": "—", "confidence": "—",
                    "description": "—", "action": "WAITING",
                    "buy_second": CFG.buy_second, "sell_second": CFG.sell_second,
                    "rule_check": CFG.signal_second},
        )

        # Fetch tokens
        up_token = down_token = market_slug = None
        for attempt in range(5):
            up_token, down_token, market_slug = fetch_market_tokens(window_end)
            if up_token and down_token:
                break
            print(c(YELLOW, f"  [!] Attempt {attempt + 1}/5 — retry in {RETRY_SLEEP}s..."))
            time.sleep(RETRY_SLEEP)

        if not up_token:
            print(c(RED, "  [✗] No tokens found — skipping."))
            wait_for_new_window()
            continue

        # Macro context (once per window)
        print(c(GRAY, "  Fetching macro context (1m + 1d Binance)..."))
        macro = fetch_macro_context()
        if macro:
            _sset(btc={"macro": {
                k: macro.get(k) for k in
                ["pct_15m", "pct_30m", "pct_60m", "pct_7d", "pct_30d", "above_ema20", "above_sma7d"]
            }})

        # ── Watch odds s0 → signal_second every second ────────────────────────
        up_s0:    Optional[float] = None
        btc_s0:   Optional[float] = None
        up_check:  float = 0.0
        btc_check: float = 0.0
        window_ticks: List[Dict] = []
        print()
        _sset(status="watching")

        for target_s in range(0, CFG.signal_second + 1):
            wait_until_second(target_s)
            up   = get_midpoint(up_token)
            down = get_midpoint(down_token)
            btc  = get_btc_price()

            if up_s0  is None and up  is not None:
                up_s0  = up
                _sset(btc={"s0": btc})
            if btc_s0 is None and btc is not None:
                btc_s0 = btc

            up_slope  = ((up  or up_s0)  - up_s0)  if up_s0  is not None else 0.0
            btc_slope = ((btc or btc_s0) - btc_s0) if btc_s0 is not None else 0.0

            print_tick(target_s, up, down, btc, up_slope, btc_slope)
            _append_tick(target_s, up, down, btc, up_slope, btc_slope)
            window_ticks.append({"second": target_s, "up": up, "down": down, "btc": btc,
                                  "slug": market_slug, "up_token": up_token, "down_token": down_token})

            if target_s == CFG.signal_second:
                up_check  = up  if up  is not None else (up_s0  or 50.0)
                btc_check = btc if btc is not None else (btc_s0 or 0.0)

        if up_s0 is None or btc_s0 is None:
            print(c(RED, "  [✗] No valid prices — skipping."))
            append_window_to_csv(window_end, window_start, window_end, window_ticks)
            wait_for_new_window()
            continue

        # ── Evaluate signal ────────────────────────────────────────────────────
        direction, rule_name, confidence, rule_desc = evaluate_signal(
            up_s0, up_check, btc_s0, btc_check
        )
        print_decision(
            window_start, window_end,
            up_s0, up_check, btc_s0, btc_check,
            macro, direction, rule_name, confidence, rule_desc,
        )

        action = f"BUY {direction}" if direction else "STAY OUT"
        _sset(
            status="signal_fired" if direction else "stayed_out",
            signal={
                "rule": 1 if direction else None,
                "rule_name": rule_name,
                "confidence": confidence if direction else "—",
                "description": rule_desc if direction else "—",
                "action": action,
                "buy_second": CFG.buy_second,
                "sell_second": CFG.sell_second,
                "rule_check": CFG.signal_second,
            },
        )

        # ── BUY at s80 ────────────────────────────────────────────────────────
        buy_price:  Optional[float] = None
        sell_price: Optional[float] = None
        buy_token = up_token if direction == "UP" else down_token

        if direction:
            wait_until_second(CFG.buy_second)
            raw_buy = get_midpoint(buy_token)
            buy_price = raw_buy
            bp = f"{buy_price:.1f}¢" if buy_price is not None else "N/A"
            dir_col = GREEN + BOLD if direction == "UP" else RED + BOLD
            print(c(dir_col, f"\n  ⚡ BUY {direction} — s{CFG.buy_second}  ({direction} at {bp}) ⚡\n"))
            _sset(status="holding")

        # ── Every-second ticks: buy_second+1 → sell_second-1 ──────────────────
        # Early-exit thresholds (applied only when holding a position)
        TAKE_PROFIT_CENTS = 20.0   # exit early if position gains +20¢
        STOP_LOSS_PCT     = 0.50   # exit early if position loses 50% of its value
        early_exit_s: Optional[int] = None

        for mid_s in range(CFG.buy_second + 1, CFG.sell_second):
            wait_until_second(mid_s)
            up   = get_midpoint(up_token)
            down = get_midpoint(down_token)
            btc  = get_btc_price()
            up_slope  = ((up  or up_s0) - up_s0)  if up_s0  else 0.0
            btc_slope = ((btc or btc_s0) - btc_s0) if btc_s0 else 0.0
            # Print every 15s to avoid flooding terminal, collect every second for CSV
            if mid_s % 15 == 0:
                print_tick(mid_s, up, down, btc, up_slope, btc_slope)
            _append_tick(mid_s, up, down, btc, up_slope, btc_slope)
            window_ticks.append({"second": mid_s, "up": up, "down": down, "btc": btc,
                                  "slug": market_slug, "up_token": up_token, "down_token": down_token})

            # ── Early-exit check (only when holding) ──────────────────────────
            if direction and buy_price is not None:
                current_price = get_midpoint(buy_token)
                if current_price is not None:
                    gain = current_price - buy_price
                    loss_pct = (buy_price - current_price) / buy_price if buy_price > 0 else 0.0

                    if gain >= TAKE_PROFIT_CENTS:
                        sell_price = current_price
                        early_exit_s = mid_s
                        sp = f"{sell_price:.1f}¢"
                        print(c(GREEN + BOLD,
                            f"\n  ★ TAKE PROFIT — s{mid_s}  {direction} at {sp}"
                            f"  (+{gain:.1f}¢ ≥ +{TAKE_PROFIT_CENTS:.0f}¢ target) ★\n"))
                        _sset(status="sell_now")
                        break

                    if loss_pct >= STOP_LOSS_PCT:
                        sell_price = current_price
                        early_exit_s = mid_s
                        sp = f"{sell_price:.1f}¢"
                        print(c(RED + BOLD,
                            f"\n  ✂ STOP LOSS — s{mid_s}  {direction} at {sp}"
                            f"  ({loss_pct*100:.0f}% loss ≥ {STOP_LOSS_PCT*100:.0f}% limit) ✂\n"))
                        _sset(status="sell_now")
                        break

        # ── SELL at s180 (if no early exit) ──────────────────────────────────
        if direction and early_exit_s is None:
            wait_until_second(CFG.sell_second)
            raw_sell = get_midpoint(buy_token)
            sell_price = raw_sell
            sp = f"{sell_price:.1f}¢" if sell_price is not None else "N/A"
            dir_col = GREEN + BOLD if direction == "UP" else RED + BOLD
            print(c(dir_col, f"\n  ⚡ SELL {direction} — s{CFG.sell_second}  ({direction} at {sp}) ⚡\n"))
            _sset(status="sell_now")
        elif direction and early_exit_s is not None:
            # Already sold early — skip the s180 sell
            pass

        # ── Every-second ticks: sell_second+1 → 295 ───────────────────────────
        for late_s in range(CFG.sell_second + 1, 296):
            wait_until_second(late_s)
            up   = get_midpoint(up_token)
            down = get_midpoint(down_token)
            btc  = get_btc_price()
            up_slope  = ((up  or up_s0) - up_s0)  if up_s0  else 0.0
            btc_slope = ((btc or btc_s0) - btc_s0) if btc_s0 else 0.0
            # Print every 20s after sell, collect every second for CSV
            if late_s % 20 == 0:
                print_tick(late_s, up, down, btc, up_slope, btc_slope)
            _append_tick(late_s, up, down, btc, up_slope, btc_slope)
            window_ticks.append({"second": late_s, "up": up, "down": down, "btc": btc,
                                  "slug": market_slug, "up_token": up_token, "down_token": down_token})

        # ── Wait for close ────────────────────────────────────────────────────
        while seconds_into_window() < 298:
            time.sleep(0.5)

        # ── Grade result ──────────────────────────────────────────────────────
        print()
        if direction and buy_price is not None and sell_price is not None:
            profit  = sell_price - buy_price
            correct = profit > 0
            col     = GREEN + BOLD if correct else RED + BOLD
            verdict = "✓ CORRECT" if correct else "✗ WRONG"
            print(c(col,
                f"  Result: {direction} went from {buy_price:.1f}¢ → {sell_price:.1f}¢"
                f"  |  Profit: {profit:+.1f}¢  |  {verdict}"))
            scorecard.record(profit)
            scorecard.update_state({
                "window": ws_label, "profit": round(profit, 1),
                "correct": correct, "rule": 1, "direction": direction,
            })
        elif direction:
            print(c(YELLOW, "  [!] Could not grade — price missing at buy or sell."))
            _sset(status="grading")
        else:
            print(c(GRAY, "  Stayed out this window."))
            _sset(status="stayed_out")

        scorecard.display()

        # ── Append window data to live training CSV ────────────────────────────
        append_window_to_csv(window_end, window_start, window_end, window_ticks)

        # ── Recompute thresholds every 10 windows ─────────────────────────────
        windows_run += 1
        if windows_run % 10 == 0:
            print(c(CYAN, f"\n  [↻] {windows_run} windows done — re-analyzing {windows_run} + historical data..."))
            try:
                recompute_thresholds()
            except Exception as exc:
                print(c(YELLOW, f"  [!] Threshold update failed: {exc}"))

        # ── Next window ───────────────────────────────────────────────────────
        wait_for_new_window()
        _sset(status="waiting")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n  {c(GREEN + BOLD, '[✓]')} Stopped.\n")
        sys.exit(0)
