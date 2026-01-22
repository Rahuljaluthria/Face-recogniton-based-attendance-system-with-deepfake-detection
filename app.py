# app.py
from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Get absolute path to the Excel file
excel_path = os.path.join(os.path.dirname(__file__), 'attendance', 'attendance.xlsx')

# Load Excel file once
students_df = pd.read_excel(excel_path)

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    name = data.get("name", "").strip().lower()
    uid = str(data.get("uid", "")).strip()

    # Filter by name (case-insensitive) and UID
    match = students_df[
        (students_df['Name'].str.lower() == name) &
        (students_df['UID'].astype(str) == uid)
    ]

    if not match.empty:
        return jsonify({"success": True})
    else:
        return jsonify({"success": False})

if __name__ == "__main__":
    app.run(port=5000)
