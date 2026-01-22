import sqlite3
import pandas as pd
import os
from datetime import datetime, date
from typing import Optional, List, Dict, Any

class AttendanceDatabase:
    def __init__(self, db_path: str = "attendance/attendance.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()
        
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Students table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                uid TEXT NOT NULL UNIQUE,
                section TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Attendance records table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                date DATE NOT NULL DEFAULT (DATE('now')),
                status TEXT NOT NULL CHECK (status IN ('Present', 'Absent')),
                marked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confidence_score REAL,
                detection_method TEXT DEFAULT 'face_recognition',
                deepfake_avg_score REAL,
                antispoof_avg_score REAL,
                combined_score REAL,
                frames_used INTEGER,
                decision TEXT,
                FOREIGN KEY (student_id) REFERENCES students (id),
                UNIQUE(student_id, date)
            )
        ''')
        
        # Face embeddings table (for future use)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                embedding_data BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def migrate_from_excel(self, excel_path: str):
        """Migrate data from Excel to SQLite"""
        if not os.path.exists(excel_path):
            print(f"Excel file {excel_path} not found!")
            return False
            
        try:
            df = pd.read_excel(excel_path)
            df.columns = df.columns.str.strip()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert students
            for _, row in df.iterrows():
                try:
                    cursor.execute('''
                        INSERT OR IGNORE INTO students (name, uid, section)
                        VALUES (?, ?, ?)
                    ''', (row['Name'], row['UID'], row['Section']))
                    
                    # Mark attendance if present in Excel
                    if pd.notna(row.get('Attendance')) and row['Attendance'] == 'P':
                        student_id = cursor.execute(
                            'SELECT id FROM students WHERE name = ?', 
                            (row['Name'],)
                        ).fetchone()
                        
                        if student_id:
                            cursor.execute('''
                                INSERT OR IGNORE INTO attendance_records 
                                (student_id, status, date)
                                VALUES (?, 'Present', DATE('now'))
                            ''', (student_id[0],))
                            
                except sqlite3.IntegrityError as e:
                    print(f"Skipping duplicate entry: {row['Name']} - {e}")
                    
            conn.commit()
            conn.close()
            print(f"✅ Successfully migrated data from {excel_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error migrating from Excel: {e}")
            return False
            
    def add_student(self, name: str, uid: str, section: str) -> bool:
        """Add a new student"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO students (name, uid, section)
                VALUES (?, ?, ?)
            ''', (name, uid, section))
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False
            
    def get_all_students(self) -> Dict[str, Dict[str, Any]]:
        """Get all students as dictionary"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, uid, section FROM students')
        
        students = {}
        for row in cursor.fetchall():
            students[row[1]] = {  # row[1] is name
                'id': row[0],
                'UID': row[2],
                'Section': row[3]
            }
            
        conn.close()
        return students
        
    def mark_attendance(self, student_name: str, confidence: float = None, 
                       detection_method: str = 'face_recognition',
                       deepfake_avg_score: float = None,
                       antispoof_avg_score: float = None,
                       combined_score: float = None,
                       frames_used: int = None,
                       decision: str = None) -> bool:
        """Mark student as present with detailed scoring information"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get student ID
            student_id = cursor.execute(
                'SELECT id FROM students WHERE name = ?', 
                (student_name,)
            ).fetchone()
            
            if not student_id:
                conn.close()
                return False
                
            # Mark attendance with detailed scores
            cursor.execute('''
                INSERT OR REPLACE INTO attendance_records 
                (student_id, status, confidence_score, detection_method,
                 deepfake_avg_score, antispoof_avg_score, combined_score,
                 frames_used, decision)
                VALUES (?, 'Present', ?, ?, ?, ?, ?, ?, ?)
            ''', (student_id[0], confidence, detection_method,
                  deepfake_avg_score, antispoof_avg_score, combined_score,
                  frames_used, decision))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error marking attendance: {e}")
            return False
            
    def get_todays_attendance(self) -> set:
        """Get list of students marked present today"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT s.name 
            FROM students s 
            JOIN attendance_records ar ON s.id = ar.student_id 
            WHERE ar.date = DATE('now') AND ar.status = 'Present'
        ''')
        
        present_students = {row[0] for row in cursor.fetchall()}
        conn.close()
        return present_students
        
    def export_to_excel(self, excel_path: str, date_filter: Optional[str] = None) -> bool:
        """Export attendance data to Excel"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            if date_filter:
                query = '''
                    SELECT s.name, s.uid, s.section, 
                           COALESCE(ar.status, 'Absent') as attendance,
                           ar.marked_at, ar.confidence_score,
                           ar.deepfake_avg_score, ar.antispoof_avg_score,
                           ar.combined_score, ar.frames_used, ar.decision,
                           ar.detection_method
                    FROM students s 
                    LEFT JOIN attendance_records ar ON s.id = ar.student_id 
                        AND ar.date = ?
                    ORDER BY s.section, s.name
                '''
                df = pd.read_sql_query(query, conn, params=(date_filter,))
            else:
                # Today's attendance
                query = '''
                    SELECT s.name, s.uid, s.section, 
                           COALESCE(ar.status, 'Absent') as attendance,
                           ar.marked_at, ar.confidence_score,
                           ar.deepfake_avg_score, ar.antispoof_avg_score,
                           ar.combined_score, ar.frames_used, ar.decision,
                           ar.detection_method
                    FROM students s 
                    LEFT JOIN attendance_records ar ON s.id = ar.student_id 
                        AND ar.date = DATE('now')
                    ORDER BY s.section, s.name
                '''
                df = pd.read_sql_query(query, conn)
                
            df.columns = ['Name', 'UID', 'Section', 'Attendance', 'Marked At', 'Confidence',
                         'Deepfake Score', 'Anti-Spoof Score', 'Combined Score', 
                         'Frames Used', 'Decision', 'Detection Method']
            
            # Convert attendance status for Excel compatibility
            df['Attendance'] = df['Attendance'].map({'Present': 'P', 'Absent': ''})
            
            # Create Excel with proper formatting
            if not os.path.isabs(excel_path):
                excel_path = os.path.abspath(excel_path)
            os.makedirs(os.path.dirname(excel_path), exist_ok=True)
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Attendance')
                
            conn.close()
            print(f"✅ Attendance exported to {excel_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error exporting to Excel: {e}")
            return False
            
    def get_attendance_summary(self, days: int = 7) -> pd.DataFrame:
        """Get attendance summary for last N days"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT s.name, s.section,
                   COUNT(CASE WHEN ar.status = 'Present' THEN 1 END) as present_days,
                   COUNT(ar.id) as total_days,
                   ROUND(COUNT(CASE WHEN ar.status = 'Present' THEN 1 END) * 100.0 / 
                         MAX(COUNT(ar.id), 1), 2) as attendance_percentage
            FROM students s 
            LEFT JOIN attendance_records ar ON s.id = ar.student_id 
                AND ar.date >= DATE('now', '-' || ? || ' days')
            GROUP BY s.id, s.name, s.section
            ORDER BY s.section, s.name
        '''
        
        df = pd.read_sql_query(query, conn, params=(days,))
        conn.close()
        return df

    def cleanup_old_records(self, days_to_keep: int = 30):
        """Clean up old attendance records"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM attendance_records 
            WHERE date < DATE('now', '-' || ? || ' days')
        ''', (days_to_keep,))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        print(f"🗑️ Cleaned up {deleted_count} old attendance records")
        return deleted_count