#!/usr/bin/env python3
"""
Monitoring script for Binance Futures Trading Bot
"""

import sqlite3
import json
from datetime import datetime, timezone
from typing import Dict, Any, List
import argparse

def get_db_connection(db_file: str):
    """Get database connection"""
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    return conn

def get_account_summary(db_file: str) -> Dict[str, Any]:
    """Get account summary from database"""
    conn = get_db_connection(db_file)
    try:
        # Get latest account snapshot
        cursor = conn.execute("""
            SELECT * FROM account_snapshot 
            ORDER BY ts DESC LIMIT 1
        """)
        latest_snapshot = cursor.fetchone()
        
        # Get daily P&L
        today = datetime.now(timezone.utc).date().isoformat()
        cursor = conn.execute("""
            SELECT SUM(profit) as daily_pnl FROM trades 
            WHERE DATE(ts) = ?
        """, (today,))
        daily_pnl_result = cursor.fetchone()
        daily_pnl = float(daily_pnl_result['daily_pnl'] or 0)
        
        # Get total P&L
        cursor = conn.execute("SELECT SUM(profit) as total_pnl FROM trades")
        total_pnl_result = cursor.fetchone()
        total_pnl = float(total_pnl_result['total_pnl'] or 0)
        
        # Get open orders
        cursor = conn.execute("""
            SELECT COUNT(*) as open_orders FROM orders 
            WHERE status IN ('NEW', 'PARTIALLY_FILLED')
        """)
        open_orders_result = cursor.fetchone()
        open_orders = int(open_orders_result['open_orders'])
        
        # Get recent trades
        cursor = conn.execute("""
            SELECT * FROM trades 
            ORDER BY ts DESC LIMIT 10
        """)
        recent_trades = [dict(row) for row in cursor.fetchall()]
        
        return {
            "latest_balance": float(latest_snapshot['balance']) if latest_snapshot else 0,
            "daily_pnl": daily_pnl,
            "total_pnl": total_pnl,
            "open_orders": open_orders,
            "recent_trades": recent_trades,
            "last_update": latest_snapshot['ts'] if latest_snapshot else None
        }
    finally:
        conn.close()

def get_risk_summary(db_file: str, start_balance: float) -> Dict[str, Any]:
    """Get risk management summary"""
    conn = get_db_connection(db_file)
    try:
        # Get current balance
        cursor = conn.execute("""
            SELECT balance FROM account_snapshot 
            ORDER BY ts DESC LIMIT 1
        """)
        result = cursor.fetchone()
        current_balance = float(result['balance']) if result else 0
        
        # Calculate daily loss
        daily_loss = start_balance - current_balance
        daily_loss_pct = (daily_loss / start_balance * 100) if start_balance > 0 else 0
        
        # Get open positions count
        cursor = conn.execute("""
            SELECT COUNT(*) as open_positions FROM orders 
            WHERE status IN ('NEW', 'PARTIALLY_FILLED')
        """)
        open_positions_result = cursor.fetchone()
        open_positions = int(open_positions_result['open_positions'])
        
        return {
            "start_balance": start_balance,
            "current_balance": current_balance,
            "daily_loss": daily_loss,
            "daily_loss_pct": daily_loss_pct,
            "open_positions": open_positions,
            "max_positions": 3,  # From config
            "daily_loss_limit_pct": 3.0,  # From config
            "risk_status": "OK" if daily_loss_pct < 3.0 else "WARNING"
        }
    finally:
        conn.close()

def print_summary(db_file: str, start_balance: float = None):
    """Print comprehensive summary"""
    print("=" * 60)
    print("BINANCE FUTURES BOT MONITOR")
    print("=" * 60)
    
    # Account summary
    account_summary = get_account_summary(db_file)
    print(f"\n📊 ACCOUNT SUMMARY:")
    print(f"   Current Balance: ${account_summary['latest_balance']:.2f}")
    print(f"   Daily P&L: ${account_summary['daily_pnl']:.2f}")
    print(f"   Total P&L: ${account_summary['total_pnl']:.2f}")
    print(f"   Open Orders: {account_summary['open_orders']}")
    
    if account_summary['last_update']:
        print(f"   Last Update: {account_summary['last_update']}")
    
    # Risk summary
    if start_balance:
        risk_summary = get_risk_summary(db_file, start_balance)
        print(f"\n⚠️  RISK SUMMARY:")
        print(f"   Start Balance: ${risk_summary['start_balance']:.2f}")
        print(f"   Daily Loss: ${risk_summary['daily_loss']:.2f} ({risk_summary['daily_loss_pct']:.2f}%)")
        print(f"   Daily Loss Limit: {risk_summary['daily_loss_limit_pct']:.1f}%")
        print(f"   Open Positions: {risk_summary['open_positions']}/{risk_summary['max_positions']}")
        print(f"   Risk Status: {risk_summary['risk_status']}")
    
    # Recent trades
    if account_summary['recent_trades']:
        print(f"\n📈 RECENT TRADES:")
        for trade in account_summary['recent_trades'][:5]:
            timestamp = trade['ts'][:19] if trade['ts'] else "N/A"
            profit = trade['profit'] or 0
            profit_color = "🟢" if profit > 0 else "🔴" if profit < 0 else "⚪"
            print(f"   {profit_color} {trade['side']} {trade['qty']} @ ${trade['price']:.2f} "
                  f"(P&L: ${profit:.2f}) - {timestamp}")
    
    print("\n" + "=" * 60)

def export_data(db_file: str, output_file: str):
    """Export data to JSON file"""
    conn = get_db_connection(db_file)
    try:
        # Get all data
        data = {}
        
        # Account snapshots
        cursor = conn.execute("SELECT * FROM account_snapshot ORDER BY ts DESC LIMIT 100")
        data['account_snapshots'] = [dict(row) for row in cursor.fetchall()]
        
        # Trades
        cursor = conn.execute("SELECT * FROM trades ORDER BY ts DESC LIMIT 100")
        data['trades'] = [dict(row) for row in cursor.fetchall()]
        
        # Orders
        cursor = conn.execute("SELECT * FROM orders ORDER BY ts DESC LIMIT 100")
        data['orders'] = [dict(row) for row in cursor.fetchall()]
        
        # Write to file
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"Data exported to {output_file}")
        
    finally:
        conn.close()

def main():
    parser = argparse.ArgumentParser(description="Monitor Binance Futures Bot")
    parser.add_argument("--db", default="async_bot_state.db", help="Database file path")
    parser.add_argument("--start-balance", type=float, help="Starting balance for risk calculation")
    parser.add_argument("--export", help="Export data to JSON file")
    
    args = parser.parse_args()
    
    if args.export:
        export_data(args.db, args.export)
    else:
        print_summary(args.db, args.start_balance)

if __name__ == "__main__":
    main()
