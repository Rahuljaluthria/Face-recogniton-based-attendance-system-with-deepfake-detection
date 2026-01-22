# app.py
from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
import os
import sys

# Add parent directory to path to import database module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import AttendanceDatabase

app = Flask(__name__)
CORS(app)

# Initialize database
db = AttendanceDatabase()

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    name = data.get("name", "").strip().lower()
    uid = str(data.get("uid", "")).strip()

    # Get student data from SQLite
    students = db.get_all_students()
    
    # Check credentials
    for student_name, student_info in students.items():
        if (student_name.lower() == name and 
            str(student_info['UID']) == uid):
            return jsonify({"success": True})
    
    return jsonify({"success": False})

@app.route("/attendance", methods=["GET"])
def get_attendance():
    """Get today's attendance"""
    try:
        # Export current attendance to Excel and return data
        excel_path = os.path.join(os.path.dirname(__file__), 'attendance', 'attendance.xlsx')
        db.export_to_excel(excel_path)
        
        # Read the exported Excel for response
        df = pd.read_excel(excel_path)
        return jsonify({
            "success": True,
            "data": df.to_dict('records'),
            "total_students": len(df),
            "present_count": len(df[df['Attendance'] == 'P'])
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/export", methods=["POST"])
def export_attendance():
    """Export attendance for specific date range"""
    try:
        data = request.json
        date_filter = data.get("date", None)  # YYYY-MM-DD format
        
        excel_path = os.path.join(os.path.dirname(__file__), 'attendance', f'attendance_export_{date_filter or "today"}.xlsx')
        success = db.export_to_excel(excel_path, date_filter)
        
        if success:
            return jsonify({"success": True, "file": excel_path})
        else:
            return jsonify({"success": False, "error": "Export failed"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/summary", methods=["GET"])
def attendance_summary():
    """Get attendance summary for last 7 days"""
    try:
        days = int(request.args.get('days', 7))
        summary_df = db.get_attendance_summary(days)
        return jsonify({
            "success": True,
            "summary": summary_df.to_dict('records'),
            "period_days": days
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(port=5000)
