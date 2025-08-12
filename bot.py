#!/usr/bin/env python3
"""
async_futures_bot.py

Async Binance USDT-M Futures bot scaffold (Testnet by default).

Features:
 - Async market data via BinanceSocketManager (kline futures socket)
 - User Data Stream via listenKey + websocket to receive ACCOUNT/ORDER events
 - ExchangeInfo-driven precision & minNotional checks
 - Leverage via API
 - Risk sizing, stop-loss and take-profit (closePosition)
 - Sqlite persistence of orders/trades/account snapshots
 - EMA crossover strategy hook (swapable)
 - Retry/backoff for REST calls

CAVEATS:
 - ALWAYS test on testnet first (TESTNET=True)
 - This is scaffold code: add robust production features (supervised monitoring, better error/reconnect logic, better order reconciliation, secure key management).
"""

import os
import asyncio
import json
import math
import aiohttp
import sqlite3
import logging
from datetime import datetime, timezone
from typing import Optional

from binance import AsyncClient, BinanceSocketManager
import pandas as pd
from ta.trend import EMAIndicator
import websockets   # used for connecting to user data stream wss
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------- CONFIG ----------
TESTNET = True  # set False ONLY after extensive testing
API_KEY = os.environ.get("BINANCE_API_KEY", "")
API_SECRET = os.environ.get("BINANCE_API_SECRET", "")

SYMBOL = "BTCUSDT"
KLINE_INTERVAL = "1m"   # 1m kline; used with BinanceSocketManager.kline_futures_socket
LEVERAGE = 5
RISK_PER_TRADE = 0.01   # fraction of balance to risk per trade (e.g., 0.01 = 1%)
STOPLOSS_PCT = 0.005    # 0.5% SL
TAKE_PROFIT_PCT = 0.01  # 1% TP
DAILY_MAX_LOSS_PCT = 0.03
DB_FILE = "async_bot_state.db"
MIN_ORDER_USD_FALLBACK = 5.0
KLINE_HISTORY = 200

# ---------- LOGGING ----------
logger = logging.getLogger("async_futures_bot")
logger.setLevel(logging.INFO)
h = logging.StreamHandler()
h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(h)

# ---------- UTIL: DB ----------
def init_db():
    c = sqlite3.connect(DB_FILE)
    cur = c.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS orders (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      symbol TEXT,
      orderId INTEGER,
      clientOrderId TEXT,
      side TEXT,
      type TEXT,
      qty REAL,
      price REAL,
      status TEXT,
      raw TEXT,
      ts TEXT
    );
    CREATE TABLE IF NOT EXISTS trades (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      symbol TEXT,
      side TEXT,
      qty REAL,
      price REAL,
      profit REAL,
      ts TEXT
    );
    CREATE TABLE IF NOT EXISTS account_snapshot (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      balance REAL,
      ts TEXT
    );
    """)
    c.commit(); c.close()

# ---------- RETRY / BACKOFF ----------
async def with_backoff(coro_fn, *a, retries=5, base_delay=1.0, **kw):
    delay = base_delay
    for attempt in range(1, retries+1):
        try:
            return await coro_fn(*a, **kw)
        except Exception as e:
            logger.warning(f"Attempt {attempt} error: {e}")
            if attempt == retries:
                raise
            await asyncio.sleep(delay)
            delay *= 2

# ---------- Exchange info helpers ----------
async def get_exchange_info(async_client: AsyncClient):
    info = await with_backoff(async_client.get_exchange_info)
    return info

def filters_for_symbol(exchange_info: dict, symbol: str):
    for s in exchange_info.get("symbols", []):
        if s["symbol"] == symbol:
            return s.get("filters", []), s
    raise RuntimeError("Symbol not found in exchangeInfo")

def precision_and_min_notional(filters):
    step_size = None
    tick_size = None
    min_notional = MIN_ORDER_USD_FALLBACK
    for f in filters:
        if f["filterType"] == "LOT_SIZE":
            step_size = float(f["stepSize"])
        if f["filterType"] == "PRICE_FILTER":
            tick_size = float(f["tickSize"])
        if f["filterType"] == "MIN_NOTIONAL":
            min_notional = float(f.get("notional", min_notional))
    def prec_from_step(step):
        if not step:
            return 8
        s = ('{:.8f}'.format(step)).rstrip('0')
        if '.' in s:
            return len(s.split('.')[1])
        return 0
    qty_prec = prec_from_step(step_size)
    price_prec = prec_from_step(tick_size)
    return qty_prec, price_prec, min_notional

def round_down(x: float, precision: int):
    q = 10 ** precision
    return math.floor(x * q) / q

# ---------- Position sizing ----------
def calc_qty(balance_usd: float, price: float, risk_pct: float, stop_pct: float, qty_precision: int, min_notional: float):
    risk_amount = balance_usd * risk_pct
    per_unit_risk = price * stop_pct
    if per_unit_risk <= 0:
        return 0.0
    raw_qty = risk_amount / per_unit_risk
    notional = raw_qty * price
    if notional < min_notional:
        logger.info(f"Notional {notional:.2f} below min_notional {min_notional:.2f} -> qty 0")
        return 0.0
    qty = round_down(raw_qty, qty_precision)
    return qty

# ---------- Persistence helpers ----------
def store_order_record(symbol, order_json):
    c = sqlite3.connect(DB_FILE); cur = c.cursor()
    cur.execute("""INSERT INTO orders(symbol,orderId,clientOrderId,side,type,qty,price,status,raw,ts)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (symbol,
                 order_json.get("orderId") or order_json.get("orderId"),
                 order_json.get("clientOrderId"),
                 order_json.get("side"),
                 order_json.get("type"),
                 float(order_json.get("origQty") or order_json.get("executedQty") or 0),
                 float(order_json.get("avgPrice") or order_json.get("price") or 0),
                 order_json.get("status"),
                 json.dumps(order_json),
                 datetime.now(timezone.utc).isoformat()))
    c.commit(); c.close()

def snapshot_account_to_db(balance):
    c = sqlite3.connect(DB_FILE); cur = c.cursor()
    cur.execute("INSERT INTO account_snapshot(balance,ts) VALUES(?,?)", (balance, datetime.now(timezone.utc).isoformat()))
    c.commit(); c.close()

# ---------- User data stream: create listenKey via REST (fapi) ----------
# We'll use aiohttp to post to the futures testnet endpoint to get listenKey
async def create_futures_listen_key(api_key: str, api_secret: str, testnet: bool = True):
    # Futures testnet base (per docs): https://testnet.binancefuture.com (REST)
    base = "https://testnet.binancefuture.com" if testnet else "https://fapi.binance.com"
    url = f"{base}/fapi/v1/listenKey"
    headers = {"X-MBX-APIKEY": api_key}
    async with aiohttp.ClientSession() as s:
        async with s.post(url, headers=headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"ListenKey create failed: {resp.status} {text}")
            res = await resp.json()
            # response: {"listenKey": "<key>"}
            return res["listenKey"]

# ---------- Strategy: EMA crossover (modular) ----------
def ema_signal_from_df(df: pd.DataFrame) -> Optional[str]:
    # Requires at least 22 rows for ema21
    if len(df) < 22:
        return None
    df = df.copy()
    df["ema9"] = EMAIndicator(df["close"], window=9).ema_indicator()
    df["ema21"] = EMAIndicator(df["close"], window=21).ema_indicator()
    prev = df.iloc[-2]
    cur  = df.iloc[-1]
    cross_up = prev["ema9"] < prev["ema21"] and cur["ema9"] > cur["ema21"]
    cross_down = prev["ema9"] > prev["ema21"] and cur["ema9"] < cur["ema21"]
    if cross_up: return "BUY"
    if cross_down: return "SELL"
    return None

# ---------- Main async bot ----------
async def run_bot():
    init_db()
    client = await AsyncClient.create(API_KEY, API_SECRET, testnet=TESTNET)
    try:
        logger.info("Connected AsyncClient")
        # get exchange info
        exchange_info = await get_exchange_info(client)
        filters, syminfo = filters_for_symbol(exchange_info, SYMBOL)
        qty_prec, price_prec, min_notional = precision_and_min_notional(filters)
        logger.info(f"Symbol precisions qty={qty_prec}, price={price_prec}, min_notional={min_notional}")

        # set leverage
        try:
            res = await with_backoff(client.futures_change_leverage, symbol=SYMBOL, leverage=LEVERAGE)
            logger.info(f"Leverage set: {res}")
        except Exception as e:
            logger.warning(f"Could not set leverage: {e}")

        # initial account snapshot
        acct = await with_backoff(client.futures_account_balance)
        balance = 0.0
        for a in acct:
            if a["asset"] == "USDT":
                balance = float(a["balance"])
                break
        snapshot_account_to_db(balance)
        start_balance = balance
        daily_max_loss = start_balance * DAILY_MAX_LOSS_PCT
        logger.info(f"Starting balance: {start_balance:.2f}, daily_max_loss: {daily_max_loss:.2f}")

        # create listenKey (user data stream) and open websocket to receive account/order updates
        listen_key = await create_futures_listen_key(API_KEY, API_SECRET, testnet=TESTNET)
        # build user data wss url (futures): wss://fstream.binancefuture.com/ws/<listenKey>  (per docs)
        wss_base = "wss://fstream.binancefuture.com/ws" if not TESTNET else "wss://stream.binancefuture.com/ws"
        user_data_ws_url = f"{wss_base}/{listen_key}"
        logger.info(f"User data websocket url prepared (listenKey OK)")

        # create BinanceSocketManager for market data (kline futures)
        bsm = BinanceSocketManager(client)

        # we'll run two tasks:
        #  1) kline listener & strategy loop (consumes kline_futures_socket)
        #  2) user data websocket listener (receives ACCOUNT_UPDATE and ORDER_TRADE_UPDATE)

        async def user_data_listener():
            logger.info("Starting user data listener")
            # reconnect logic is intentionally simple here
            while True:
                try:
                    async with websockets.connect(user_data_ws_url) as ws:
                        logger.info("Connected to user data websocket")
                        async for msg in ws:
                            try:
                                data = json.loads(msg)
                                # store snapshot for ACCOUNT_UPDATE events
                                if data.get("e") == "ACCOUNT_UPDATE":
                                    # write snapshot
                                    logger.debug("ACCOUNT_UPDATE received")
                                    # conservative: take wallet balance again via REST
                                    acct2 = await with_backoff(client.futures_account_balance)
                                    bal = 0.0
                                    for a in acct2:
                                        if a["asset"] == "USDT":
                                            bal = float(a["balance"]); break
                                    snapshot_account_to_db(bal)
                                if data.get("e") == "ORDER_TRADE_UPDATE":
                                    logger.info("ORDER_TRADE_UPDATE: " + json.dumps(data))
                                    # store order event in DB (raw)
                                    c = sqlite3.connect(DB_FILE); cur = c.cursor()
                                    cur.execute("INSERT INTO orders(symbol,orderId,clientOrderId,side,type,qty,price,status,raw,ts) VALUES(?,?,?,?,?,?,?,?,?,?)",
                                                (data.get("o", {}).get("s"),
                                                 data.get("o", {}).get("i"),
                                                 data.get("o", {}).get("c"),
                                                 data.get("o", {}).get("S"),
                                                 data.get("o", {}).get("o"),
                                                 float(data.get("o", {}).get("q") or 0),
                                                 float(data.get("o", {}).get("p") or 0),
                                                 data.get("o", {}).get("X"),
                                                 json.dumps(data),
                                                 datetime.now(timezone.utc).isoformat()))
                                    c.commit(); c.close()
                            except Exception as ee:
                                logger.exception("Error processing user-data message")
                except Exception as e:
                    logger.exception("User data websocket crashed; reconnecting in 5s")
                    await asyncio.sleep(5)

        async def kline_strategy_listener():
            logger.info("Starting kline strategy listener")
            # kline_futures_socket provided by BinanceSocketManager (Async)
            async with bsm.kline_futures_socket(symbol=SYMBOL, interval=KLINE_INTERVAL) as stream:
                while True:
                    try:
                        res = await stream.recv()
                        # res format: kline stream payload; we extract kline close
                        k = res.get("k", {})
                        is_closed = k.get("x", False)
                        if not is_closed:
                            # only act on closed candles to avoid noise
                            continue
                        # fetch last KLINE_HISTORY candles REST to build dataframe (reliable)
                        raw_klines = await with_backoff(client.futures_klines, symbol=SYMBOL, interval=KLINE_INTERVAL, limit=KLINE_HISTORY)
                        df = pd.DataFrame(raw_klines, columns=[
                            "open_time","open","high","low","close","volume","close_time","qav",
                            "num_trades","taker_base_vol","taker_quote_vol","ignore"
                        ])
                        df["close"] = df["close"].astype(float)
                        signal = ema_signal_from_df(df)
                        logger.info(f"Signal: {signal}")
                        
                        # check daily loss cap
                        acct_bal = 0.0
                        try:
                            acct_info = await with_backoff(client.futures_account_balance)
                            for a in acct_info:
                                if a["asset"] == "USDT":
                                    acct_bal = float(a["balance"]); break
                            if start_balance - acct_bal >= daily_max_loss:
                                logger.warning("Daily loss threshold reached; stopping strategy loop")
                                return
                        except Exception as e:
                            logger.error(f"Failed to check daily loss: {e}")
                            continue
                        
                        if signal:
                            price = float(df["close"].iloc[-1])
                            qty = calc_qty(acct_bal, price, RISK_PER_TRADE, STOPLOSS_PCT, qty_prec, min_notional)
                            if qty <= 0:
                                logger.info("qty <= 0 after sizing; skipping trade")
                                continue
                            # place market order
                            side = signal
                            logger.info(f"Placing MARKET {side} qty={qty} price={price}")
                            order_res = await with_backoff(client.futures_create_order, symbol=SYMBOL, side=side, type="MARKET", quantity=qty)
                            store_order_record(SYMBOL, order_res)
                            # place stop-loss & take-profit (closePosition)
                            if side == "BUY":
                                stop_price = format(price * (1 - STOPLOSS_PCT), f".{price_prec}f")
                                tp_price = format(price * (1 + TAKE_PROFIT_PCT), f".{price_prec}f")
                                close_side = "SELL"
                            else:
                                stop_price = format(price * (1 + STOPLOSS_PCT), f".{price_prec}f")
                                tp_price = format(price * (1 - TAKE_PROFIT_PCT), f".{price_prec}f")
                                close_side = "BUY"
                            logger.info(f"Placing SL at {stop_price} and TP at {tp_price}")
                            # STOP_MARKET (closePosition True) and TAKE_PROFIT_MARKET (closePosition True)
                            try:
                                sl_res = await with_backoff(client.futures_create_order,
                                                           symbol=SYMBOL,
                                                           side=close_side,
                                                           type="STOP_MARKET",
                                                           stopPrice=stop_price,
                                                           closePosition=True)
                                tp_res = await with_backoff(client.futures_create_order,
                                                           symbol=SYMBOL,
                                                           side=close_side,
                                                           type="TAKE_PROFIT_MARKET",
                                                           stopPrice=tp_price,
                                                           closePosition=True)
                                logger.info(f"SL order: {sl_res}, TP order: {tp_res}")
                            except Exception as e:
                                logger.exception("Failed to place SL/TP orders")
                    except Exception:
                        logger.exception("kline_strategy_listener error; sleeping briefly")
                        await asyncio.sleep(1)

        # Prepare precision variables captured earlier
        qty_prec, price_prec, min_notional = qty_prec, price_prec, min_notional

        # Create tasks
        tasks = [
            asyncio.create_task(user_data_listener()),
            asyncio.create_task(kline_strategy_listener())
        ]
        # run until tasks complete (kline may exit on loss clamp)
        await asyncio.gather(*tasks)
    finally:
        await client.close_connection()
        logger.info("Closed AsyncClient")

# ---------- Entrypoint ----------
if __name__ == "__main__":
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Exiting.")

