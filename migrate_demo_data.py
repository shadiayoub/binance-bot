#!/usr/bin/env python3
"""
Demo to Live Data Migration Script
Migrates demo trading data to live mode database
"""

import sqlite3
import shutil
import os
from datetime import datetime

def migrate_demo_to_live():
    """Migrate demo data to live database"""
    
    demo_db = "demo_bot_state.db"
    live_db = "live_bot_state.db"
    
    if not os.path.exists(demo_db):
        print("❌ No demo database found")
        return False
    
    print("🔄 Migrating demo data to live mode...")
    
    try:
        # Create backup of live database if it exists
        if os.path.exists(live_db):
            backup_name = f"live_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            shutil.copy2(live_db, backup_name)
            print(f"📦 Created backup: {backup_name}")
        
        # Copy demo database to live
        shutil.copy2(demo_db, live_db)
        print("✅ Demo data migrated successfully")
        
        # Update balance records to reflect live mode
        with sqlite3.connect(live_db) as conn:
            # Clear demo balance records
            conn.execute("DELETE FROM account_snapshot WHERE balance = 100.0")
            print("🧹 Cleaned demo balance records")
            
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False

def clear_demo_data():
    """Clear demo database for fresh start"""
    
    demo_db = "demo_bot_state.db"
    
    if os.path.exists(demo_db):
        os.remove(demo_db)
        print("🗑️ Demo database cleared")
    else:
        print("ℹ️ No demo database to clear")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "migrate":
            migrate_demo_to_live()
        elif sys.argv[1] == "clear":
            clear_demo_data()
        else:
            print("Usage: python migrate_demo_data.py [migrate|clear]")
    else:
        print("Demo Data Migration Tool")
        print("Usage:")
        print("  python migrate_demo_data.py migrate  # Migrate demo to live")
        print("  python migrate_demo_data.py clear    # Clear demo data")
