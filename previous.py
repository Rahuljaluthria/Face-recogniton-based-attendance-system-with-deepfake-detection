import cv2
import numpy as np
import face_recognition
import os
import pandas as pd
from PIL import Image  # For saving face crop
from Facelog.deepfake_detector import load_model, detect_deepfake  # Deepfake tools

# Load deepfake model
deepfake_model_path = "Facelog/models/best_deepfake_detector.pth"
deepfake_model = load_model(deepfake_model_path)

# Load student details
excel_path = "attendance/attendance.xlsx"
if os.path.exists(excel_path):
    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip()
    if "Attendance" not in df.columns:
        df["Attendance"] = ""
    student_data = {row["Name"]: {"UID": row["UID"], "Section": row["Section"]} for _, row in df.iterrows()}
    print("✅ Loaded student details from attendance.xlsx")
else:
    print("[❌] attendance.xlsx not found!")
    student_data = {}

# Load known faces
known_faces_dir = "faces"
known_face_encodings, known_face_names = [], []

for filename in os.listdir(known_faces_dir):
    img_path = os.path.join(known_faces_dir, filename)
    try:
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)

        if not encodings:
            print(f"⚠️ No face found in {filename}, skipping...")
            continue

        known_face_encodings.append(encodings[0])
        known_face_names.append(os.path.splitext(filename)[0])
    except Exception as e:
        print(f"[❌] Error loading {filename}: {e}")

if not known_face_encodings:
    print("[❌] No valid face encodings found. Exiting...")
    exit()

print(f"✅ Loaded {len(known_face_encodings)} known faces")
print("Loaded face names:", known_face_names)

# Attendance tracker
marked_students = set(df[df["Attendance"] == "P"]["Name"])

# Webcam
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("[❌] No camera found. Exiting...")
    exit()

video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
video_capture.set(cv2.CAP_PROP_FPS, 30)

frame_skip = 5
frame_count = 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("[❌] Failed to grab frame. Exiting...")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")

    if not face_locations:
        continue

    print(f"🔍 Detected {len(face_locations)} face(s) in frame")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.45)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        detected_name = "Unknown"
        if face_distances.size > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                detected_name = known_face_names[best_match_index]

        if detected_name in student_data and detected_name not in marked_students:
            # Extract and save face for deepfake detection
            face_crop = rgb_frame[top:bottom, left:right]
            face_image = Image.fromarray(face_crop)
            temp_path = "temp_face.jpg"
            face_image.save(temp_path)

            # Run deepfake check
            result, confidence = detect_deepfake(temp_path, deepfake_model)
            print(f"🔎 Deepfake detection: {result} ({confidence:.2f}% confidence)")

            # Mark attendance if real and confident
            if result == "REAL" and confidence > 5:
                df["Attendance"] = df["Attendance"].astype(object)  # Fix dtype warning
                df.loc[df["Name"] == detected_name, "Attendance"] = "P"
                df.to_excel(excel_path, index=False)
                marked_students.add(detected_name)
                print(f"✅ Marked {detected_name} as present in attendance.xlsx")
            else:
                print(f"⚠️ {detected_name} flagged as {result} ({confidence:.2f}%). Not marked.")

        # Show info
        student_info = student_data.get(detected_name, {"UID": "", "Section": ""})
        info_text = f"{detected_name} | UID: {student_info['UID']} | Sec: {student_info['Section']}"
        text_color = (0, 255, 0) if detected_name != "Unknown" else (0, 0, 255)

        cv2.rectangle(frame, (left, top), (right, bottom), text_color, 2)
        cv2.putText(frame, info_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    cv2.imshow("FaceLog - Attendance System", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        df["Attendance"] = ""
        df.to_excel(excel_path, index=False)
        marked_students.clear()
        print("🔄 Attendance reset!")

video_capture.release()
cv2.destroyAllWindows()
