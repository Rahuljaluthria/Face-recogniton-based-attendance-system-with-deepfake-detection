
🧠 FaceLog – AI-Based Face Recognition Attendance System with Deepfake Detection

FaceLog is a smart attendance system that leverages InsightFace for real-time face recognition and integrates deepfake detection to ensure secure and reliable attendance logging. It uses Python for the core logic and offers a beautiful desktop interface built with Electron.js.

---

🚀 Features

✅ Real-time face recognition using InsightFace  
✅ Deepfake detection to prevent spoofing attacks  
✅ Marks late arrivals if detected 20+ minutes after class starts  
✅ Automatically logs attendance in an Excel sheet  
✅ Monthly reset and support for multiple student sections  
✅ Powered by GPU (CUDA) for faster inference  
✅ Desktop UI built with Electron.js  
🚧 Flutter mobile app in progress

---

🛠️ Tech Stack

- Python
  - InsightFace (buffalo_l)
  - OpenCV
  - Pandas
  - Sklearn (for cosine similarity)
  - PyTorch (for deepfake detection)
- Electron.js (for frontend GUI)
- Excel (via pandas for attendance logs)
- Flutter (mobile version coming soon)

---

📁 Project Structure

FaceLog/
|
├── faces/                      Folder containing reference student images
├── attendance/
│   └── attendance.xlsx         Excel file to store attendance logs
├── Facelog/
│   ├── deepfake_detector.py    Deepfake detection module
│   └── models/
│       └── best_deepfake_detector.pth
├── insightface_models/         Cached InsightFace models
├── main.py                     Core Python script to run the system
├── electron-app/               Electron.js frontend (optional)
└── README.md

---

⚙️ How It Works

1. Loads all known student faces and creates embeddings using InsightFace.
2. Starts the webcam and matches faces using cosine similarity.
3. For each match:
   - Extracts the detected face
   - Runs deepfake detection
   - If real and not yet marked, marks the student as 'P' (Present)' in the Excel sheet
4. Displays the name and bounding box on the webcam feed
5. Prevents duplicate entries and resets attendance monthly

---

📦 Installation & Setup

1️⃣ Clone the Repository
git clone https://github.com/yourusername/FaceLog.git
cd FaceLog

2️⃣ Install Python Dependencies
pip install -r requirements.txt

Make sure you have:
- InsightFace
- OpenCV
- torch / torchvision
- pandas
- scikit-learn
- PIL

3️⃣ Prepare Folders
- Put student face images in faces/ folder (named as Name.jpg)
- Create attendance/attendance.xlsx with columns: Name, UID, Section, Attendance

4️⃣ Run the System
python main.py

Press q to stop the webcam.

---

💻 Electron Frontend (Optional)

To start the Electron frontend:
cd electron-app
npm install
npm start

Ensure Python backend is running first.

---

🧪 Deepfake Detection

The Facelog/deepfake_detector.py module uses a pretrained PyTorch model to classify each detected face as REAL or FAKE. Only real faces are logged in the attendance.

---


🧠 Future Enhancements

- Flutter-based mobile version
- Email/SMS notifications for attendance
- Database integration (PostgreSQL or MongoDB)
- Admin panel with real-time analytics

---

📜 License

This project is licensed under the MIT License.

---

🙌 Credits

Built with 💻, ☕, and passion by Rahul.

---

🤝 Contributing

Feel free to fork the repo and open a pull request. Bug fixes, enhancements, and suggestions are always welcome!

---

FaceLog – Making Attendance Smarter and Safer.
