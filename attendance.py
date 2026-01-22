import pandas as pd
import os

# Define the path for the attendance file
attendance_file = "attendance/attendance.xlsx"

# Create folder if it doesn't exist
os.makedirs("attendance", exist_ok=True)

# Define initial columns
columns = ["Student Name", "Date", "Time", "Status"]

# Check if file exists, if not, create it
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=columns)
    df.to_excel(attendance_file, index=False)
    print("✅ attendance.xlsx created successfully!")
else:
    print("📄 attendance.xlsx already exists.")
