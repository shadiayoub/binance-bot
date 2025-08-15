#!/usr/bin/env python3
"""
Multi-Symbol Binance Futures Trading Bot with ML Predictions
Supports trading multiple cryptocurrencies simultaneously
"""

import asyncio
import logging
import signal
import sys
import time
import os
from typing import Dict, List, Optional, Any
from datetime import datetime

from binance import AsyncClient
from binance.streams import BinanceSocketManager
from binance.exceptions import BinanceAPIException

from config import Config
from database import DatabaseManager
from risk_manager import RiskManager
from enhanced_strategy import EnhancedStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('multi_symbol_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MultiSymbolBot:
    """Multi-symbol trading bot that can trade multiple cryptocurrencies"""
    
    def __init__(self):
        self.config = Config()
        # Use appropriate database based on trading mode
        db_file = self.config.get_db_file()
        self.db = DatabaseManager(db_file)
        self.risk_manager = RiskManager(self.config, self.db)
        self.strategy = EnhancedStrategy(self.config.__dict__)
        
        # Multi-symbol tracking
        self.symbols = self.config.SYMBOLS
        self.symbol_data = {}  # Store data for each symbol
        self.symbol_strategies = {}  # Store strategy for each symbol
        self.symbol_positions = {}  # Track positions for each symbol
        
        # Client and socket manager
        self.client = None
        self.bsm = None
        self.listen_key = None
        
        # Shutdown flag
        self.shutdown_event = asyncio.Event()
        
        # Initialize symbol data
        for symbol in self.symbols:
            self.symbol_data[symbol] = {
                'klines': [],
                'current_price': 0.0,
                'precision': {'qty': 5, 'price': 2},
                'min_notional': 5.0
            }
            self.symbol_strategies[symbol] = EnhancedStrategy(self.config.__dict__)
            self.symbol_positions[symbol] = None
            
            # Load ML model for this symbol
            try:
                model_path = f"models/{symbol}_gradient_boosting.joblib"
                if os.path.exists(model_path):
                    self.symbol_strategies[symbol].initialize_predictor()
                    self.symbol_strategies[symbol].predictor.load_model(model_path)
                    logger.info(f"ML model loaded for {symbol}")
                else:
                    logger.warning(f"No ML model found for {symbol} at {model_path}")
            except Exception as e:
                logger.error(f"Failed to load ML model for {symbol}: {e}")
    
    async def initialize(self):
        """Initialize the multi-symbol bot"""
        logger.info("Initializing Multi-Symbol Binance Futures Bot...")
        
        try:
            # Initialize client
            self.client = await AsyncClient.create(
                api_key=self.config.API_KEY,
                api_secret=self.config.API_SECRET,
                testnet=self.config.TESTNET
            )
            
            # Initialize socket manager
            self.bsm = BinanceSocketManager(self.client)
            
            # Setup exchange info for all symbols
            await self._setup_exchange_info()
            
            # Set leverage for all symbols
            await self._set_leverage_all()
            
            # Get initial balance
            await self._get_initial_balance()
            
            # Create listen key for user data
            await self._create_listen_key()
            
            logger.info("Multi-symbol bot initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize bot: {e}")
            return False
    
    async def _setup_exchange_info(self):
        """Setup exchange info for all symbols"""
        try:
            exchange_info = await self.client.futures_exchange_info()
            
            for symbol in self.symbols:
                symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
                if symbol_info:
                    # Get precision and min notional
                    qty_precision = 5
                    price_precision = 2
                    min_notional = 5.0
                    
                    for filter_info in symbol_info['filters']:
                        if filter_info['filterType'] == 'LOT_SIZE':
                            qty_precision = len(str(float(filter_info['stepSize'])).split('.')[-1].rstrip('0'))
                        elif filter_info['filterType'] == 'PRICE_FILTER':
                            price_precision = len(str(float(filter_info['tickSize'])).split('.')[-1].rstrip('0'))
                        elif filter_info['filterType'] == 'MIN_NOTIONAL':
                            min_notional = float(filter_info['notional'])
                    
                    self.symbol_data[symbol]['precision'] = {
                        'qty': qty_precision,
                        'price': price_precision
                    }
                    self.symbol_data[symbol]['min_notional'] = min_notional
                    
                    logger.info(f"Symbol {symbol} precision - Qty: {qty_precision}, Price: {price_precision}, Min notional: {min_notional}")
                else:
                    logger.warning(f"Symbol {symbol} not found in exchange info")
                    
        except Exception as e:
            logger.error(f"Failed to setup exchange info: {e}")
    
    async def _set_leverage_all(self):
        """Set leverage for all symbols"""
        for symbol in self.symbols:
            try:
                config = self.config.SYMBOL_CONFIGS.get(symbol, {})
                leverage = config.get('leverage', self.config.LEVERAGE)
                
                result = await self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
                logger.info(f"Leverage set for {symbol}: {result}")
                
            except Exception as e:
                logger.error(f"Failed to set leverage for {symbol}: {e}")
    
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

    async def _get_initial_balance(self):
        """Get initial account balance"""
        try:
            account_balance = await self._with_backoff(self.client.futures_account_balance)
            balance = 0.0
            for asset in account_balance:
                if asset['asset'] == 'USDT':
                    balance = float(asset['balance'])
                    break
            self.risk_manager.start_balance = balance
            logger.info(f"Initial balance: {balance:.2f} USDT")
            return True
        except Exception as e:
            logger.error(f"Failed to get initial balance: {e}")
            logger.warning("Using demo balance for testing")
            self.risk_manager.start_balance = 100.0  # Demo balance
            logger.info("Demo balance set: 100.00 USDT")
            return False
    
    async def _create_listen_key(self):
        """Create listen key for user data stream"""
        try:
            result = await self.client.futures_stream_get_listen_key()
            # Handle different response formats
            if isinstance(result, dict) and 'listenKey' in result:
                self.listen_key = result['listenKey']
            elif isinstance(result, str):
                self.listen_key = result
            else:
                logger.warning("Unexpected listen key response format")
                self.listen_key = None
                return
            logger.info("Listen key created successfully")
        except Exception as e:
            logger.warning(f"Failed to create listen key: {e}")
            self.listen_key = None
    
    async def _kline_listener(self, symbol: str):
        """Listen to kline data for a specific symbol"""
        logger.info(f"Starting kline listener for {symbol}")
        
        while not self.shutdown_event.is_set():
            try:
                # Use the context manager approach which should work with older versions
                async with self.bsm.kline_futures_socket(symbol=symbol, interval=self.config.KLINE_INTERVAL) as stream:
                    logger.info(f"Connected to kline stream for {symbol}")
                    
                    while not self.shutdown_event.is_set():
                        try:
                            # Add timeout to prevent hanging
                            result = await asyncio.wait_for(stream.recv(), timeout=60.0)
                            
                            # Debug: Log all incoming messages for first few seconds
                            if len(self.symbol_data[symbol]['klines']) < 3:
                                logger.info(f"[{symbol}] Raw message: {result}")
                            
                            if result and 'e' in result and (result['e'] == 'kline' or result['e'] == 'continuous_kline'):
                                kline = result['k']
                                
                                # Debug: Log first few klines to confirm data is flowing
                                if len(self.symbol_data[symbol]['klines']) < 5:
                                    logger.info(f"[{symbol}] Received kline: {kline['c']} at {kline['t']}")
                                
                                # Update symbol data
                                kline_data = {
                                    'open_time': kline['t'],
                                    'open': float(kline['o']),
                                    'high': float(kline['h']),
                                    'low': float(kline['l']),
                                    'close': float(kline['c']),
                                    'volume': float(kline['v']),
                                    'close_time': kline['T']
                                }
                                
                                self.symbol_data[symbol]['klines'].append(kline_data)
                                self.symbol_data[symbol]['current_price'] = float(kline['c'])
                                
                                # Keep only recent klines
                                if len(self.symbol_data[symbol]['klines']) > self.config.KLINE_HISTORY:
                                    self.symbol_data[symbol]['klines'] = self.symbol_data[symbol]['klines'][-self.config.KLINE_HISTORY:]
                                
                                                            # Generate signal if we have enough data
                            if len(self.symbol_data[symbol]['klines']) >= 50:
                                await self._process_symbol_signal(symbol)
                            else:
                                # Debug logging to show data collection progress
                                if len(self.symbol_data[symbol]['klines']) % 10 == 0:  # Log every 10 klines
                                    logger.info(f"[{symbol}] Collected {len(self.symbol_data[symbol]['klines'])} klines, need 50")
                                
                                # Heartbeat every 5 minutes to show bot is alive
                                if len(self.symbol_data[symbol]['klines']) % 5 == 0 and len(self.symbol_data[symbol]['klines']) > 0:
                                    logger.info(f"[{symbol}] Heartbeat: {len(self.symbol_data[symbol]['klines'])} klines collected, latest price: {self.symbol_data[symbol]['current_price']}")
                                
                        except asyncio.CancelledError:
                            logger.info(f"Kline listener cancelled for {symbol}")
                            break
                        except Exception as e:
                            logger.error(f"Error receiving kline data for {symbol}: {e}")
                            break  # Break inner loop to reconnect
                            
            except Exception as e:
                logger.error(f"Failed to connect to kline stream for {symbol}: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _process_symbol_signal(self, symbol: str):
        """Process trading signal for a specific symbol"""
        try:
            # Create DataFrame for strategy
            import pandas as pd
            df = pd.DataFrame(self.symbol_data[symbol]['klines'])
            
            # Get current price
            current_price = self.symbol_data[symbol]['current_price']
            
            # Generate signal using symbol-specific strategy
            signal, strength, signal_data = self.symbol_strategies[symbol].generate_signal(df)
            
            # Log signal information
            ml_prediction = signal_data.get('ml_prediction', 0.0)
            ml_confidence = signal_data.get('ml_confidence', 0.0)
            technical_signals = signal_data.get('technical_signals', {})
            
            logger.info(f"[{symbol}] Price: {current_price:.2f}, Signal: {signal}, Strength: {strength:.2f}")
            logger.info(f"[{symbol}] ML Prediction: {ml_prediction:.4f}, Confidence: {ml_confidence:.2%}")
            logger.info(f"[{symbol}] Technical signals - EMA: {technical_signals.get('ema_signal', 0.00):.2f}, "
                       f"RSI: {technical_signals.get('rsi_signal', 0.00):.2f}, "
                       f"BB: {technical_signals.get('bb_signal', 0.00):.2f}")
            
            # Execute trade if signal is strong enough and trading is enabled
            if signal and strength > self.config.MIN_SIGNAL_STRENGTH:
                if self.config.ENABLE_LIVE_TRADING:
                    logger.info(f"[{symbol}] 🚀 EXECUTING LIVE TRADE: {signal} @ {current_price:.2f} (Strength: {strength:.2f})")
                    await self._execute_trade(symbol, signal, strength, current_price)
                else:
                    logger.info(f"[{symbol}] 📊 PAPER TRADE: {signal} @ {current_price:.2f} (Strength: {strength:.2f}) - Live trading disabled")
            elif signal and strength > 0.1:
                logger.info(f"[{symbol}] ⚠️ Signal too weak: {signal} (Strength: {strength:.2f} < {self.config.MIN_SIGNAL_STRENGTH})")
                
        except Exception as e:
            logger.error(f"Error processing signal for {symbol}: {e}")
    
    async def _execute_trade(self, symbol: str, signal: str, strength: float, price: float):
        """Execute trade for a specific symbol"""
        try:
            # Safety check: Ensure trading is enabled
            if not self.config.ENABLE_LIVE_TRADING:
                logger.warning(f"[{symbol}] Live trading is disabled, skipping trade")
                return
            
            # Check if we already have a position
            if self.symbol_positions[symbol]:
                logger.info(f"[{symbol}] Already have position, skipping trade")
                return
            
            # Additional safety checks
            if strength < self.config.MIN_SIGNAL_STRENGTH:
                logger.warning(f"[{symbol}] Signal strength {strength:.2f} below minimum {self.config.MIN_SIGNAL_STRENGTH}")
                return
            
            # Get symbol-specific config
            config = self.config.SYMBOL_CONFIGS.get(symbol, {})
            min_quantity = config.get('min_quantity', 0.001)
            
            # Get current balance and validate
            balance = await self._get_current_balance()
            if balance < self.config.MIN_BALANCE_THRESHOLD:
                logger.warning(f"[{symbol}] Balance {balance:.2f} below minimum threshold {self.config.MIN_BALANCE_THRESHOLD}")
                return
            
            logger.info(f"[{symbol}] Current balance: {balance:.2f} USDT")
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                balance, 
                price,
                config.get('risk_per_trade', self.config.RISK_PER_TRADE),
                config.get('stoploss_pct', self.config.STOPLOSS_PCT),
                self.symbol_data[symbol]['precision']['qty'],
                min_quantity
            )
            
            if position_size <= 0:
                logger.warning(f"[{symbol}] Position size too small: {position_size}")
                return
            
            # Place the trade
            side = "BUY" if signal == "BUY" else "SELL"
            quantity = self._round_down(position_size, self.symbol_data[symbol]['precision']['qty'])
            
            order = await self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            
            logger.info(f"[{symbol}] {side} order placed: {quantity} @ {price:.2f}")
            
            # Store position info
            self.symbol_positions[symbol] = {
                'side': side,
                'quantity': quantity,
                'entry_price': price,
                'order_id': order['orderId'],
                'timestamp': datetime.now()
            }
            
            # Place stop loss and take profit
            await self._place_stop_orders(symbol, side, quantity, price, config)
            
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
    
    async def _place_stop_orders(self, symbol: str, side: str, quantity: float, price: float, config: dict):
        """Place stop loss and take profit orders"""
        try:
            stoploss_pct = config.get('stoploss_pct', self.config.STOPLOSS_PCT)
            take_profit_pct = config.get('take_profit_pct', self.config.TAKE_PROFIT_PCT)
            
            if side == "BUY":
                stop_price = price * (1 - stoploss_pct)
                take_profit_price = price * (1 + take_profit_pct)
            else:
                stop_price = price * (1 + stoploss_pct)
                take_profit_price = price * (1 - take_profit_pct)
            
            # Round prices
            stop_price = self._round_down(stop_price, self.symbol_data[symbol]['precision']['price'])
            take_profit_price = self._round_down(take_profit_price, self.symbol_data[symbol]['precision']['price'])
            
            # Place stop loss
            await self.client.futures_create_order(
                symbol=symbol,
                side="SELL" if side == "BUY" else "BUY",
                type="STOP_MARKET",
                quantity=quantity,
                stopPrice=stop_price
            )
            
            # Place take profit
            await self.client.futures_create_order(
                symbol=symbol,
                side="SELL" if side == "BUY" else "BUY",
                type="TAKE_PROFIT_MARKET",
                quantity=quantity,
                stopPrice=take_profit_price
            )
            
            logger.info(f"[{symbol}] Stop orders placed - SL: {stop_price:.2f}, TP: {take_profit_price:.2f}")
            
        except Exception as e:
            logger.error(f"Error placing stop orders for {symbol}: {e}")
    
    async def _get_current_balance(self) -> float:
        """Get current account balance"""
        try:
            account_balance = await self._with_backoff(self.client.futures_account_balance)
            balance = 0.0
            for asset in account_balance:
                if asset['asset'] == 'USDT':
                    balance = float(asset['balance'])
                    break
            return balance
        except Exception as e:
            logger.error(f"Failed to get current balance: {e}")
            return 0.0
    
    def _round_down(self, value: float, precision: int) -> float:
        """Round down to specified precision"""
        factor = 10 ** precision
        return float(int(value * factor) / factor)
    
    async def _user_data_listener(self):
        """Listen to user data stream for order updates"""
        if not self.listen_key:
            logger.warning("No listen key available, skipping user data listener")
            return
            
        try:
            # Try different method signatures
            try:
                async with self.bsm.futures_user_socket(self.listen_key) as stream:
                    logger.info("Connected to user data websocket")
            except TypeError:
                # Fallback for different API versions
                async with self.bsm.futures_user_socket() as stream:
                    logger.info("Connected to user data websocket (fallback)")
                
                while not self.shutdown_event.is_set():
                    try:
                        result = await stream.recv()
                        
                        if result['e'] == 'ORDER_TRADE_UPDATE':
                            order_data = result['o']
                            symbol = order_data['s']
                            status = order_data['X']
                            
                            if status in ['FILLED', 'CANCELED', 'EXPIRED']:
                                # Clear position if order is filled
                                if status == 'FILLED' and self.symbol_positions.get(symbol):
                                    self.symbol_positions[symbol] = None
                                    logger.info(f"[{symbol}] Position closed")
                                
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        logger.error(f"Error in user data listener: {e}")
                        await asyncio.sleep(1)
                        
        except Exception as e:
            logger.error(f"Failed to start user data listener: {e}")
    
    async def run(self):
        """Run the multi-symbol bot"""
        try:
            # Start user data listener only if we have a listen key
            if self.listen_key:
                asyncio.create_task(self._user_data_listener())
            else:
                logger.info("Skipping user data listener (no listen key available)")
            
            # Start kline listeners for all symbols
            kline_tasks = []
            for symbol in self.symbols:
                task = asyncio.create_task(self._kline_listener(symbol))
                kline_tasks.append(task)
            
            # Wait for shutdown
            await self.shutdown_event.wait()
            
            # Cancel all tasks
            for task in kline_tasks:
                task.cancel()
            
            logger.info("Multi-symbol bot shutdown complete")
            
        except Exception as e:
            logger.error(f"Error in main run loop: {e}")
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.listen_key:
                await self.client.futures_stream_close_listen_key(self.listen_key)
            
            if self.client:
                await self.client.close_connection()
                
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

async def main():
    """Main function"""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and run bot
    bot = MultiSymbolBot()
    
    try:
        if await bot.initialize():
            await bot.run()
        else:
            logger.error("Failed to initialize bot")
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await bot.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
