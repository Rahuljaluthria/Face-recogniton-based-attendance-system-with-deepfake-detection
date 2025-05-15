import pandas as pd

# Path to the attendance file
excel_path = "attendance/attendance.xlsx"

# Load the Excel file
df = pd.read_excel(excel_path)

# Clear the "Attendance" column
df["Attendance"] = ""

# Save the updated file
df.to_excel(excel_path, index=False)

print("✅ Attendance has been reset successfully!")
