#!/usr/bin/env python3
"""
Database management for Binance Futures Trading Bot
"""

import sqlite3
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_file: str):
        self.db_file = db_file
        self.init_db()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
        except Exception as e:
            logger.error(f"Database error: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def init_db(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    orderId INTEGER,
                    clientOrderId TEXT,
                    side TEXT NOT NULL,
                    type TEXT NOT NULL,
                    qty REAL NOT NULL,
                    price REAL,
                    status TEXT NOT NULL,
                    raw TEXT,
                    ts TEXT NOT NULL
                );
                
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL NOT NULL,
                    price REAL NOT NULL,
                    profit REAL,
                    ts TEXT NOT NULL
                );
                
                CREATE TABLE IF NOT EXISTS account_snapshot (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    balance REAL NOT NULL,
                    equity REAL,
                    margin_balance REAL,
                    unrealized_pnl REAL,
                    ts TEXT NOT NULL
                );
                
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    mark_price REAL,
                    unrealized_pnl REAL,
                    liquidation_price REAL,
                    ts TEXT NOT NULL
                );
                
                CREATE TABLE IF NOT EXISTS bot_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)
            
            # Create indexes separately
            conn.executescript("""
                CREATE INDEX IF NOT EXISTS idx_orders_symbol_orderid ON orders(symbol, orderId);
                CREATE INDEX IF NOT EXISTS idx_trades_symbol_ts ON trades(symbol, ts);
                CREATE INDEX IF NOT EXISTS idx_account_snapshot_ts ON account_snapshot(ts);
                CREATE INDEX IF NOT EXISTS idx_positions_symbol_ts ON positions(symbol, ts);
            """)
            conn.commit()
    
    def store_order(self, symbol: str, order_data: Dict[str, Any]):
        """Store order record"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO orders(symbol, orderId, clientOrderId, side, type, qty, price, status, raw, ts)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                order_data.get("orderId"),
                order_data.get("clientOrderId"),
                order_data.get("side"),
                order_data.get("type"),
                float(order_data.get("origQty") or order_data.get("executedQty") or 0),
                float(order_data.get("avgPrice") or order_data.get("price") or 0),
                order_data.get("status"),
                json.dumps(order_data),
                datetime.now(timezone.utc).isoformat()
            ))
            conn.commit()
    
    def store_trade(self, symbol: str, trade_data: Dict[str, Any]):
        """Store trade record"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO trades(symbol, side, qty, price, profit, ts)
                VALUES(?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                trade_data.get("side"),
                float(trade_data.get("qty") or 0),
                float(trade_data.get("price") or 0),
                float(trade_data.get("profit") or 0),
                datetime.now(timezone.utc).isoformat()
            ))
            conn.commit()
    
    def snapshot_account(self, balance: float, equity: float = None, 
                        margin_balance: float = None, unrealized_pnl: float = None):
        """Store account snapshot"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO account_snapshot(balance, equity, margin_balance, unrealized_pnl, ts)
                VALUES(?, ?, ?, ?, ?)
            """, (
                balance,
                equity,
                margin_balance,
                unrealized_pnl,
                datetime.now(timezone.utc).isoformat()
            ))
            conn.commit()
    
    def get_daily_pnl(self, symbol: str = None) -> float:
        """Get daily P&L"""
        today = datetime.now(timezone.utc).date().isoformat()
        with self.get_connection() as conn:
            if symbol:
                cursor = conn.execute("""
                    SELECT SUM(profit) as total_pnl FROM trades 
                    WHERE symbol = ? AND DATE(ts) = ?
                """, (symbol, today))
            else:
                cursor = conn.execute("""
                    SELECT SUM(profit) as total_pnl FROM trades 
                    WHERE DATE(ts) = ?
                """, (today,))
            
            result = cursor.fetchone()
            return float(result['total_pnl'] or 0)
    
    def get_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """Get open orders"""
        with self.get_connection() as conn:
            if symbol:
                cursor = conn.execute("""
                    SELECT * FROM orders WHERE status IN ('NEW', 'PARTIALLY_FILLED') AND symbol = ?
                    ORDER BY ts DESC
                """, (symbol,))
            else:
                cursor = conn.execute("""
                    SELECT * FROM orders WHERE status IN ('NEW', 'PARTIALLY_FILLED')
                    ORDER BY ts DESC
                """)
            
            return [dict(row) for row in cursor.fetchall()]
    
    def update_bot_state(self, key: str, value: Any):
        """Update bot state"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO bot_state(key, value, updated_at)
                VALUES(?, ?, ?)
            """, (
                key,
                json.dumps(value),
                datetime.now(timezone.utc).isoformat()
            ))
            conn.commit()
    
    def get_bot_state(self, key: str) -> Optional[Any]:
        """Get bot state"""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT value FROM bot_state WHERE key = ?", (key,))
            result = cursor.fetchone()
            if result:
                return json.loads(result['value'])
            return None
