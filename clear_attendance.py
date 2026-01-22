import pandas as pd
from database import AttendanceDatabase
from datetime import date

# Initialize database
db = AttendanceDatabase()
excel_path = "attendance/attendance.xlsx"

print("🗑️ Clearing today's attendance records...")

# Clear today's attendance from SQLite
import sqlite3
conn = sqlite3.connect(db.db_path)
cursor = conn.cursor()

# Delete today's attendance records
cursor.execute("DELETE FROM attendance_records WHERE date = DATE('now')")
deleted_count = cursor.rowcount

conn.commit()
conn.close()

print(f"✅ Cleared {deleted_count} attendance records for today")

# Export updated data to Excel
db.export_to_excel(excel_path)
print(f"✅ Updated Excel file: {excel_path}")

print("✅ Attendance has been reset successfully!")
