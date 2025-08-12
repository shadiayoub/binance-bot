#!/usr/bin/env python3
"""
Risk Management for Binance Futures Trading Bot
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RiskLimits:
    max_daily_loss_pct: float = 0.03  # 3%
    max_open_positions: int = 3
    min_balance_threshold: float = 10.0
    max_position_size_pct: float = 0.1  # 10% of balance
    max_leverage: int = 10
    min_risk_reward_ratio: float = 1.5

class RiskManager:
    def __init__(self, config: Dict[str, Any], db_manager):
        self.config = config
        self.db = db_manager
        self.limits = RiskLimits()
        self.start_balance = None
        self.daily_loss_limit = None
        
    def initialize(self, start_balance: float):
        """Initialize risk manager with starting balance"""
        self.start_balance = start_balance
        self.daily_loss_limit = start_balance * self.limits.max_daily_loss_pct
        logger.info(f"Risk manager initialized - Start balance: {start_balance:.2f}, "
                   f"Daily loss limit: {self.daily_loss_limit:.2f}")
    
    def check_daily_loss_limit(self, current_balance: float) -> bool:
        """Check if daily loss limit has been reached"""
        if not self.start_balance:
            return True
        
        daily_loss = self.start_balance - current_balance
        if daily_loss >= self.daily_loss_limit:
            logger.warning(f"Daily loss limit reached: {daily_loss:.2f} >= {self.daily_loss_limit:.2f}")
            return False
        return True
    
    def check_balance_threshold(self, current_balance: float) -> bool:
        """Check if balance is above minimum threshold"""
        if current_balance < self.limits.min_balance_threshold:
            logger.warning(f"Balance below threshold: {current_balance:.2f} < {self.limits.min_balance_threshold:.2f}")
            return False
        return True
    
    def check_open_positions_limit(self, open_positions: List[Dict[str, Any]]) -> bool:
        """Check if number of open positions is within limit"""
        if len(open_positions) >= self.limits.max_open_positions:
            logger.warning(f"Too many open positions: {len(open_positions)} >= {self.limits.max_open_positions}")
            return False
        return True
    
    def calculate_position_size(self, balance: float, price: float, risk_pct: float, 
                              stop_pct: float, qty_precision: int, min_notional: float) -> float:
        """Calculate safe position size based on risk parameters"""
        # Basic risk-based sizing
        risk_amount = balance * risk_pct
        per_unit_risk = price * stop_pct
        
        if per_unit_risk <= 0:
            logger.warning("Invalid stop percentage - per unit risk <= 0")
            return 0.0
        
        raw_qty = risk_amount / per_unit_risk
        notional = raw_qty * price
        
        # Check minimum notional
        if notional < min_notional:
            logger.info(f"Notional {notional:.2f} below minimum {min_notional:.2f}")
            return 0.0
        
        # Check maximum position size
        max_position_notional = balance * self.limits.max_position_size_pct
        if notional > max_position_notional:
            logger.info(f"Position size reduced from {notional:.2f} to {max_position_notional:.2f}")
            notional = max_position_notional
            raw_qty = notional / price
        
        # Round down to precision
        qty = self._round_down(raw_qty, qty_precision)
        return qty
    
    def validate_trade_parameters(self, side: str, qty: float, price: float, 
                                stop_price: float, take_profit_price: float) -> bool:
        """Validate trade parameters before execution"""
        if qty <= 0:
            logger.warning("Invalid quantity: qty <= 0")
            return False
        
        if price <= 0:
            logger.warning("Invalid price: price <= 0")
            return False
        
        # Check risk-reward ratio
        if side == "BUY":
            risk = price - stop_price
            reward = take_profit_price - price
        else:
            risk = stop_price - price
            reward = price - take_profit_price
        
        if risk <= 0 or reward <= 0:
            logger.warning("Invalid stop loss or take profit levels")
            return False
        
        risk_reward_ratio = reward / risk
        if risk_reward_ratio < self.limits.min_risk_reward_ratio:
            logger.warning(f"Risk-reward ratio too low: {risk_reward_ratio:.2f} < {self.limits.min_risk_reward_ratio}")
            return False
        
        return True
    
    def check_market_conditions(self, symbol: str, current_price: float, 
                              recent_prices: List[float]) -> bool:
        """Check if market conditions are suitable for trading"""
        if len(recent_prices) < 20:
            logger.warning("Insufficient price history for market condition check")
            return False
        
        # Simple volatility check
        price_changes = [abs(recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1] 
                        for i in range(1, len(recent_prices))]
        avg_volatility = sum(price_changes) / len(price_changes)
        
        # Skip trading if volatility is too high (>5% average)
        if avg_volatility > 0.05:
            logger.info(f"Market volatility too high: {avg_volatility:.4f} > 0.05")
            return False
        
        return True
    
    def _round_down(self, x: float, precision: int) -> float:
        """Round down to specified precision"""
        import math
        q = 10 ** precision
        return math.floor(x * q) / q
    
    def get_risk_summary(self, current_balance: float, open_positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get current risk summary"""
        daily_loss = self.start_balance - current_balance if self.start_balance else 0
        daily_loss_pct = (daily_loss / self.start_balance * 100) if self.start_balance else 0
        
        return {
            "current_balance": current_balance,
            "start_balance": self.start_balance,
            "daily_loss": daily_loss,
            "daily_loss_pct": daily_loss_pct,
            "daily_loss_limit": self.daily_loss_limit,
            "open_positions": len(open_positions),
            "max_positions": self.limits.max_open_positions,
            "balance_ok": self.check_balance_threshold(current_balance),
            "daily_loss_ok": self.check_daily_loss_limit(current_balance),
            "positions_ok": self.check_open_positions_limit(open_positions)
        }
