#!/usr/bin/env python3
"""Show attendance table"""

from database import AttendanceDatabase
import sqlite3
import pandas as pd

def show_todays_attendance():
    """Show today's attendance in table format with detailed scores"""
    db = AttendanceDatabase()
    
    query = '''
        SELECT 
            s.name as "Student Name",
            s.uid as "UID", 
            s.section as "Section",
            COALESCE(ar.status, 'Absent') as "Status",
            COALESCE(strftime('%H:%M:%S', ar.marked_at), '-') as "Time",
            COALESCE(ROUND(ar.confidence_score, 1), '-') as "Confidence",
            COALESCE(ROUND(ar.deepfake_avg_score, 1), '-') as "Deepfake",
            COALESCE(ROUND(ar.antispoof_avg_score, 1), '-') as "Anti-Spoof",
            COALESCE(ar.frames_used, '-') as "Frames",
            COALESCE(ar.decision, '-') as "Decision"
        FROM students s 
        LEFT JOIN attendance_records ar ON s.id = ar.student_id 
            AND ar.date = DATE('now')
        ORDER BY s.section, s.name
    '''
    
    conn = sqlite3.connect(db.db_path)
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print("\n📋 TODAY'S ATTENDANCE TABLE WITH DETAILED SCORES")
    print("=" * 120)
    print(df.to_string(index=False))
    
    # Summary stats
    total_students = len(df)
    present_students = len(df[df['Status'] == 'Present'])
    absent_students = total_students - present_students
    attendance_rate = (present_students / total_students * 100) if total_students > 0 else 0
    
    print("\n" + "=" * 120)
    print("📊 SUMMARY")
    print("=" * 120)
    print(f"Total Students:    {total_students}")
    print(f"Present:           {present_students}")
    print(f"Absent:            {absent_students}")
    print(f"Attendance Rate:   {attendance_rate:.1f}%")
    
    # Show detailed score statistics for present students
    if present_students > 0:
        present_df = df[df['Status'] == 'Present']
        avg_deepfake = present_df[present_df['Deepfake'] != '-']['Deepfake'].astype(float).mean() if any(present_df['Deepfake'] != '-') else 0
        avg_antispoof = present_df[present_df['Anti-Spoof'] != '-']['Anti-Spoof'].astype(float).mean() if any(present_df['Anti-Spoof'] != '-') else 0
        avg_confidence = present_df[present_df['Confidence'] != '-']['Confidence'].astype(float).mean() if any(present_df['Confidence'] != '-') else 0
        
        print(f"\n📈 DETECTION QUALITY (Present Students Only)")
        print("=" * 60)
        print(f"Average Deepfake Score:     {avg_deepfake:.1f}%")
        print(f"Average Anti-Spoof Score:   {avg_antispoof:.1f}%") 
        print(f"Average Combined Score:     {avg_confidence:.1f}%")

def show_section_wise():
    """Show section-wise breakdown"""
    db = AttendanceDatabase()
    
    query = '''
        SELECT 
            s.section as "Section",
            COUNT(s.id) as "Total",
            COUNT(ar.id) as "Present",
            (COUNT(s.id) - COUNT(ar.id)) as "Absent",
            ROUND(COUNT(ar.id) * 100.0 / COUNT(s.id), 1) as "Attendance %"
        FROM students s 
        LEFT JOIN attendance_records ar ON s.id = ar.student_id 
            AND ar.date = DATE('now')
        GROUP BY s.section
        ORDER BY s.section
    '''
    
    conn = sqlite3.connect(db.db_path)
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print("\n📊 SECTION-WISE ATTENDANCE")
    print("=" * 60)
    print(df.to_string(index=False))

if __name__ == "__main__":
    show_todays_attendance()
    show_section_wise()