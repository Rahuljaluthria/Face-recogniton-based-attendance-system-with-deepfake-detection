import os

# Set custom model path to avoid permission issues
os.environ["INSIGHTFACE_HOME"] = "D:/Facerecognitonbasedattendancesystem/insightface_models"
insightface_model_dir = os.environ["INSIGHTFACE_HOME"]

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# Import InsightFace *after* setting ENV variable
from insightface.app import FaceAnalysis
from Facelog.deepfake_detector import load_model, detect_deepfake

# Initialize InsightFace app with custom root
app = FaceAnalysis(name='buffalo_l', root=insightface_model_dir, providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0)

# Deepfake model (already GPU-enabled)
deepfake_model_path = "Facelog/models/best_deepfake_detector.pth"
deepfake_model = load_model(deepfake_model_path)

# Load attendance sheet
excel_path = "attendance/attendance.xlsx"
if os.path.exists(excel_path):
    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip()
    if "Attendance" not in df.columns:
        df["Attendance"] = ""
    student_data = {row["Name"]: {"UID": row["UID"], "Section": row["Section"]} for _, row in df.iterrows()}
else:
    print("[❌] attendance.xlsx not found!")
    student_data = {}

# Load known faces and compute embeddings
known_embeddings, known_names = [], []
for file in os.listdir("faces"):
    path = os.path.join("faces", file)
    name = os.path.splitext(file)[0]
    img = cv2.imread(path)
    if img is None:
        print(f"[❌] Failed to load {file}")
        continue
    faces = app.get(img)
    if faces:
        known_embeddings.append(faces[0].embedding)
        known_names.append(name)
    else:
        print(f"⚠️ No face found in {file}")

# Start webcam
cap = cv2.VideoCapture(0)
marked_students = set(df[df["Attendance"] == "P"]["Name"])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)
    if not faces:
        continue  # Skip if no face detected

    for face in faces:
        embedding = face.embedding.reshape(1, -1)
        similarities = cosine_similarity(embedding, np.array(known_embeddings))[0]
        best_match_index = np.argmax(similarities)
        similarity_score = similarities[best_match_index]

        detected_name = "Unknown"
        if similarity_score > 0.5:
            detected_name = known_names[best_match_index]

        # Deepfake check
        if detected_name in student_data and detected_name not in marked_students:
            x1, y1, x2, y2 = map(int, face.bbox)
            face_img = frame[y1:y2, x1:x2]

            if face_img is not None and face_img.size != 0:
                temp_path = "temp_face.jpg"
                cv2.imwrite(temp_path, face_img)

                result, confidence = detect_deepfake(temp_path, deepfake_model)
                print(f"Deepfake Check for {detected_name}: {result} ({confidence}%)")

                if result == "REAL" and confidence > 38:
                    df.loc[df["Name"] == detected_name, "Attendance"] = "P"
                    df.to_excel(excel_path, index=False)
                    marked_students.add(detected_name)
                    print(f"✅ Marked {detected_name} as Present")
                else:
                    print(f"⚠️ {detected_name} flagged as {result} ({confidence}%)")
            else:
                print(f"[❌] Face image for {detected_name} is empty. Skipping deepfake detection.")

        # Draw box and name
        (x1, y1, x2, y2) = map(int, face.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) if detected_name != "Unknown" else (0, 0, 255), 2)
        cv2.putText(frame, detected_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("FaceLog - GPU Powered", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
