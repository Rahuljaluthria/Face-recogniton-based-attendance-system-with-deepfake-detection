#!/usr/bin/env python3
"""
Database migration script to add detailed scoring columns
"""

import sqlite3
import os

def migrate_database():
    """Add new columns to existing attendance_records table"""
    db_path = "attendance/attendance.db"
    
    if not os.path.exists(db_path):
        print("❌ Database not found, no migration needed")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if new columns already exist
    cursor.execute("PRAGMA table_info(attendance_records)")
    columns = [row[1] for row in cursor.fetchall()]
    
    new_columns = [
        'deepfake_avg_score',
        'antispoof_avg_score', 
        'combined_score',
        'frames_used',
        'decision'
    ]
    
    migrations_needed = [col for col in new_columns if col not in columns]
    
    if not migrations_needed:
        print("✅ Database already has all required columns")
        conn.close()
        return
    
    print(f"🔧 Adding {len(migrations_needed)} new columns to database...")
    
    try:
        # Add new columns
        for column in migrations_needed:
            if column in ['deepfake_avg_score', 'antispoof_avg_score', 'combined_score']:
                cursor.execute(f"ALTER TABLE attendance_records ADD COLUMN {column} REAL")
            elif column == 'frames_used':
                cursor.execute(f"ALTER TABLE attendance_records ADD COLUMN {column} INTEGER")
            elif column == 'decision':
                cursor.execute(f"ALTER TABLE attendance_records ADD COLUMN {column} TEXT")
            
            print(f"  ✅ Added column: {column}")
        
        conn.commit()
        print("✅ Database migration completed successfully!")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        conn.rollback()
    
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_database()