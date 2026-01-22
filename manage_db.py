#!/usr/bin/env python3
"""
Database Management and Report Generation Utility
Provides command-line tools for managing the attendance database
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from database import AttendanceDatabase

def main():
    parser = argparse.ArgumentParser(description='Attendance Database Management')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export attendance to Excel')
    export_parser.add_argument('--date', type=str, help='Date in YYYY-MM-DD format (default: today)')
    export_parser.add_argument('--output', type=str, help='Output Excel file path')
    
    # Summary command
    summary_parser = subparsers.add_parser('summary', help='Show attendance summary')
    summary_parser.add_argument('--days', type=int, default=7, help='Number of days to analyze (default: 7)')
    
    # Add student command
    add_parser = subparsers.add_parser('add-student', help='Add new student')
    add_parser.add_argument('--name', type=str, required=True, help='Student name')
    add_parser.add_argument('--uid', type=str, required=True, help='Student UID')
    add_parser.add_argument('--section', type=str, required=True, help='Student section')
    
    # Migrate command
    migrate_parser = subparsers.add_parser('migrate', help='Migrate from Excel to SQLite')
    migrate_parser.add_argument('--excel', type=str, required=True, help='Excel file path')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean old records')
    cleanup_parser.add_argument('--days', type=int, default=30, help='Keep records for N days (default: 30)')
    
    # List students command
    list_parser = subparsers.add_parser('list-students', help='List all students')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize database
    db = AttendanceDatabase()
    
    if args.command == 'export':
        output_path = args.output or f"attendance_export_{args.date or 'today'}.xlsx"
        success = db.export_to_excel(output_path, args.date)
        if success:
            print(f"✅ Attendance exported to {output_path}")
        else:
            print(f"❌ Export failed")
            
    elif args.command == 'summary':
        summary_df = db.get_attendance_summary(args.days)
        print(f"\n📊 Attendance Summary (Last {args.days} days)")
        print("=" * 60)
        print(summary_df.to_string(index=False))
        
    elif args.command == 'add-student':
        success = db.add_student(args.name, args.uid, args.section)
        if success:
            print(f"✅ Added student: {args.name}")
        else:
            print(f"❌ Failed to add student (may already exist)")
            
    elif args.command == 'migrate':
        if not os.path.exists(args.excel):
            print(f"❌ Excel file not found: {args.excel}")
            return
        success = db.migrate_from_excel(args.excel)
        if success:
            print("✅ Migration completed")
        else:
            print("❌ Migration failed")
            
    elif args.command == 'cleanup':
        deleted = db.cleanup_old_records(args.days)
        print(f"🗑️ Cleaned up {deleted} old records")
        
    elif args.command == 'list-students':
        students = db.get_all_students()
        print(f"\n👥 Registered Students ({len(students)} total)")
        print("=" * 60)
        for name, info in students.items():
            print(f"{name:<25} {info['UID']:<15} {info['Section']}")

if __name__ == "__main__":
    main()