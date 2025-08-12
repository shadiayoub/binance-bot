#!/usr/bin/env python3
"""
Improved Binance Futures Trading Bot

Enhanced version with better error handling, risk management, and modularity.
"""

import os
import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import aiohttp
import pandas as pd
from binance import AsyncClient, BinanceSocketManager
from ta.trend import EMAIndicator
import websockets

from config import Config
from database import DatabaseManager
from risk_manager import RiskManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("improved_bot")

class BinanceFuturesBot:
    def __init__(self):
        self.config = Config()
        self.config.validate()
        
        self.db = DatabaseManager(self.config.DB_FILE)
        self.risk_manager = RiskManager(self.config.__dict__, self.db)
        
        self.client: Optional[AsyncClient] = None
        self.bsm: Optional[BinanceSocketManager] = None
        self.listen_key: Optional[str] = None
        
        # Trading state
        self.symbol_info = {}
        self.qty_precision = 8
        self.price_precision = 8
        self.min_notional = 5.0
        self.start_balance = 0.0
        self.is_running = False
        
        # Graceful shutdown
        self.shutdown_event = asyncio.Event()
        
    async def initialize(self):
        """Initialize the bot"""
        logger.info("Initializing Binance Futures Bot...")
        
        # Create async client
        self.client = await AsyncClient.create(
            self.config.API_KEY, 
            self.config.API_SECRET, 
            testnet=self.config.TESTNET
        )
        
        # Get exchange info
        await self._setup_exchange_info()
        
        # Set leverage
        await self._set_leverage()
        
        # Get initial balance
        await self._get_initial_balance()
        
        # Create listen key for user data stream
        await self._create_listen_key()
        
        # Create socket manager
        self.bsm = BinanceSocketManager(self.client)
        
        logger.info("Bot initialized successfully")
    
    async def _setup_exchange_info(self):
        """Setup exchange information and precision"""
        try:
            exchange_info = await self._with_backoff(self.client.get_exchange_info)
            filters, symbol_info = self._get_filters_for_symbol(exchange_info, self.config.SYMBOL)
            self.symbol_info = symbol_info
            
            self.qty_precision, self.price_precision, self.min_notional = \
                self._get_precision_and_min_notional(filters)
            
            logger.info(f"Symbol precision - Qty: {self.qty_precision}, "
                       f"Price: {self.price_precision}, Min notional: {self.min_notional}")
        except Exception as e:
            logger.error(f"Failed to setup exchange info: {e}")
            raise
    
    async def _set_leverage(self):
        """Set leverage for the symbol"""
        try:
            result = await self._with_backoff(
                self.client.futures_change_leverage,
                symbol=self.config.SYMBOL,
                leverage=self.config.LEVERAGE
            )
            logger.info(f"Leverage set to {self.config.LEVERAGE}: {result}")
        except Exception as e:
            logger.warning(f"Could not set leverage: {e}")
    
    async def _get_initial_balance(self):
        """Get initial account balance"""
        try:
            account = await self._with_backoff(self.client.futures_account_balance)
            for asset in account:
                if asset["asset"] == "USDT":
                    self.start_balance = float(asset["balance"])
                    break
            
            self.risk_manager.initialize(self.start_balance)
            self.db.snapshot_account(self.start_balance)
            
            logger.info(f"Initial balance: {self.start_balance:.2f} USDT")
        except Exception as e:
            logger.error(f"Failed to get initial balance: {e}")
            raise
    
    async def _create_listen_key(self):
        """Create listen key for user data stream"""
        try:
            base_url = "https://testnet.binancefuture.com" if self.config.TESTNET else "https://fapi.binance.com"
            url = f"{base_url}/fapi/v1/listenKey"
            headers = {"X-MBX-APIKEY": self.config.API_KEY}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise RuntimeError(f"ListenKey creation failed: {response.status} {text}")
                    
                    result = await response.json()
                    self.listen_key = result["listenKey"]
                    logger.info("Listen key created successfully")
        except Exception as e:
            logger.error(f"Failed to create listen key: {e}")
            raise
    
    async def _with_backoff(self, coro_fn, *args, retries=5, base_delay=1.0, **kwargs):
        """Execute coroutine with exponential backoff"""
        delay = base_delay
        for attempt in range(1, retries + 1):
            try:
                return await coro_fn(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Attempt {attempt} failed: {e}")
                if attempt == retries:
                    raise
                await asyncio.sleep(delay)
                delay *= 2
    
    def _get_filters_for_symbol(self, exchange_info: dict, symbol: str):
        """Get filters for a specific symbol"""
        for s in exchange_info.get("symbols", []):
            if s["symbol"] == symbol:
                return s.get("filters", []), s
        raise RuntimeError(f"Symbol {symbol} not found in exchange info")
    
    def _get_precision_and_min_notional(self, filters):
        """Extract precision and minimum notional from filters"""
        step_size = None
        tick_size = None
        min_notional = self.config.MIN_ORDER_USD_FALLBACK
        
        for f in filters:
            if f["filterType"] == "LOT_SIZE":
                step_size = float(f["stepSize"])
            elif f["filterType"] == "PRICE_FILTER":
                tick_size = float(f["tickSize"])
            elif f["filterType"] == "MIN_NOTIONAL":
                min_notional = float(f.get("notional", min_notional))
        
        def precision_from_step(step):
            if not step:
                return 8
            s = ('{:.8f}'.format(step)).rstrip('0')
            if '.' in s:
                return len(s.split('.')[1])
            return 0
        
        qty_prec = precision_from_step(step_size)
        price_prec = precision_from_step(tick_size)
        return qty_prec, price_prec, min_notional
    
    def _round_down(self, x: float, precision: int):
        """Round down to specified precision"""
        import math
        q = 10 ** precision
        return math.floor(x * q) / q
    
    def _ema_signal(self, df: pd.DataFrame) -> Optional[str]:
        """Generate EMA crossover signal"""
        if len(df) < 22:
            return None
        
        df = df.copy()
        df["ema9"] = EMAIndicator(df["close"], window=9).ema_indicator()
        df["ema21"] = EMAIndicator(df["close"], window=21).ema_indicator()
        
        prev = df.iloc[-2]
        curr = df.iloc[-1]
        
        cross_up = prev["ema9"] < prev["ema21"] and curr["ema9"] > curr["ema21"]
        cross_down = prev["ema9"] > prev["ema21"] and curr["ema9"] < curr["ema21"]
        
        if cross_up:
            return "BUY"
        elif cross_down:
            return "SELL"
        return None
    
    async def _get_current_balance(self) -> float:
        """Get current account balance"""
        try:
            account = await self._with_backoff(self.client.futures_account_balance)
            for asset in account:
                if asset["asset"] == "USDT":
                    return float(asset["balance"])
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get current balance: {e}")
            return 0.0
    
    async def _get_open_positions(self) -> list:
        """Get current open positions"""
        try:
            positions = await self._with_backoff(self.client.futures_position_information)
            return [p for p in positions if float(p["positionAmt"]) != 0]
        except Exception as e:
            logger.error(f"Failed to get open positions: {e}")
            return []
    
    async def _place_trade(self, signal: str, price: float) -> bool:
        """Place a trade with stop loss and take profit"""
        try:
            # Get current balance and check risk limits
            current_balance = await self._get_current_balance()
            open_positions = await self._get_open_positions()
            
            if not self.risk_manager.check_daily_loss_limit(current_balance):
                logger.warning("Daily loss limit reached, skipping trade")
                return False
            
            if not self.risk_manager.check_balance_threshold(current_balance):
                logger.warning("Balance below threshold, skipping trade")
                return False
            
            if not self.risk_manager.check_open_positions_limit(open_positions):
                logger.warning("Too many open positions, skipping trade")
                return False
            
            # Calculate position size
            qty = self.risk_manager.calculate_position_size(
                current_balance, price, self.config.RISK_PER_TRADE,
                self.config.STOPLOSS_PCT, self.qty_precision, self.min_notional
            )
            
            if qty <= 0:
                logger.info("Position size too small, skipping trade")
                return False
            
            # Calculate stop loss and take profit prices
            if signal == "BUY":
                stop_price = price * (1 - self.config.STOPLOSS_PCT)
                take_profit_price = price * (1 + self.config.TAKE_PROFIT_PCT)
                close_side = "SELL"
            else:
                stop_price = price * (1 + self.config.STOPLOSS_PCT)
                take_profit_price = price * (1 - self.config.TAKE_PROFIT_PCT)
                close_side = "BUY"
            
            # Validate trade parameters
            if not self.risk_manager.validate_trade_parameters(
                signal, qty, price, stop_price, take_profit_price
            ):
                logger.warning("Invalid trade parameters")
                return False
            
            # Place market order
            logger.info(f"Placing {signal} order: qty={qty:.6f}, price={price:.2f}")
            order_result = await self._with_backoff(
                self.client.futures_create_order,
                symbol=self.config.SYMBOL,
                side=signal,
                type="MARKET",
                quantity=qty
            )
            
            self.db.store_order(self.config.SYMBOL, order_result)
            logger.info(f"Market order placed: {order_result}")
            
            # Place stop loss and take profit orders
            await self._place_stop_orders(close_side, stop_price, take_profit_price)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to place trade: {e}")
            return False
    
    async def _place_stop_orders(self, close_side: str, stop_price: float, take_profit_price: float):
        """Place stop loss and take profit orders"""
        try:
            stop_price_str = format(stop_price, f".{self.price_precision}f")
            take_profit_price_str = format(take_profit_price, f".{self.price_precision}f")
            
            # Place stop loss order
            sl_result = await self._with_backoff(
                self.client.futures_create_order,
                symbol=self.config.SYMBOL,
                side=close_side,
                type="STOP_MARKET",
                stopPrice=stop_price_str,
                closePosition=True
            )
            
            # Place take profit order
            tp_result = await self._with_backoff(
                self.client.futures_create_order,
                symbol=self.config.SYMBOL,
                side=close_side,
                type="TAKE_PROFIT_MARKET",
                stopPrice=take_profit_price_str,
                closePosition=True
            )
            
            self.db.store_order(self.config.SYMBOL, sl_result)
            self.db.store_order(self.config.SYMBOL, tp_result)
            
            logger.info(f"Stop orders placed - SL: {sl_result}, TP: {tp_result}")
            
        except Exception as e:
            logger.error(f"Failed to place stop orders: {e}")
    
    async def _user_data_listener(self):
        """Listen to user data stream"""
        logger.info("Starting user data listener")
        
        wss_base = "wss://fstream.binancefuture.com/ws"
        user_data_url = f"{wss_base}/{self.listen_key}"
        
        while not self.shutdown_event.is_set():
            try:
                async with websockets.connect(user_data_url) as ws:
                    logger.info("Connected to user data websocket")
                    
                    async for message in ws:
                        if self.shutdown_event.is_set():
                            break
                        
                        try:
                            data = json.loads(message)
                            
                            if data.get("e") == "ACCOUNT_UPDATE":
                                logger.debug("Account update received")
                                current_balance = await self._get_current_balance()
                                self.db.snapshot_account(current_balance)
                            
                            elif data.get("e") == "ORDER_TRADE_UPDATE":
                                logger.info(f"Order update: {data}")
                                self.db.store_order(
                                    data.get("o", {}).get("s"),
                                    data
                                )
                                
                        except Exception as e:
                            logger.error(f"Error processing user data message: {e}")
                            
            except Exception as e:
                logger.error(f"User data websocket error: {e}")
                if not self.shutdown_event.is_set():
                    await asyncio.sleep(5)
    
    async def _kline_strategy_listener(self):
        """Listen to kline data and execute strategy"""
        logger.info("Starting kline strategy listener")
        
        async with self.bsm.kline_futures_socket(
            symbol=self.config.SYMBOL, 
            interval=self.config.KLINE_INTERVAL
        ) as stream:
            while not self.shutdown_event.is_set():
                try:
                    result = await stream.recv()
                    kline = result.get("k", {})
                    
                    # Only process closed candles
                    if not kline.get("x", False):
                        continue
                    
                    # Get historical klines for analysis
                    klines = await self._with_backoff(
                        self.client.futures_klines,
                        symbol=self.config.SYMBOL,
                        interval=self.config.KLINE_INTERVAL,
                        limit=self.config.KLINE_HISTORY
                    )
                    
                    # Create DataFrame
                    df = pd.DataFrame(klines, columns=[
                        "open_time", "open", "high", "low", "close", "volume",
                        "close_time", "qav", "num_trades", "taker_base_vol",
                        "taker_quote_vol", "ignore"
                    ])
                    df["close"] = df["close"].astype(float)
                    
                    # Generate signal
                    signal = self._ema_signal(df)
                    current_price = float(df["close"].iloc[-1])
                    
                    logger.info(f"Current price: {current_price:.2f}, Signal: {signal}")
                    
                    # Check market conditions
                    recent_prices = df["close"].tail(20).tolist()
                    if not self.risk_manager.check_market_conditions(
                        self.config.SYMBOL, current_price, recent_prices
                    ):
                        logger.info("Market conditions not suitable for trading")
                        continue
                    
                    # Execute trade if signal exists
                    if signal:
                        await self._place_trade(signal, current_price)
                    
                except Exception as e:
                    logger.error(f"Kline strategy listener error: {e}")
                    if not self.shutdown_event.is_set():
                        await asyncio.sleep(1)
    
    async def run(self):
        """Run the bot"""
        try:
            await self.initialize()
            self.is_running = True
            
            # Create tasks
            tasks = [
                asyncio.create_task(self._user_data_listener()),
                asyncio.create_task(self._kline_strategy_listener())
            ]
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
            # Cancel tasks
            for task in tasks:
                task.cancel()
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Bot runtime error: {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up...")
        self.is_running = False
        
        if self.client:
            await self.client.close_connection()
        
        logger.info("Cleanup completed")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

async def main():
    """Main entry point"""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    bot = BinanceFuturesBot()
    
    try:
        await bot.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Main error: {e}")
    finally:
        await bot.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
