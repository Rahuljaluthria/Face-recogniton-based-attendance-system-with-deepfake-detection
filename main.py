import os
import torch

# Set custom model path to avoid permission issues
os.environ["INSIGHTFACE_HOME"] = "D:/Facerecognitonbasedattendancesystem/insightface_models"
insightface_model_dir = os.environ["INSIGHTFACE_HOME"]

# GPU optimization settings
if torch.cuda.is_available():
    print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"💾 Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    # Enable optimizations for faster processing
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
else:
    print("💻 Running on CPU mode")

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# Import InsightFace *after* setting ENV variable
from insightface.app import FaceAnalysis
from Facelog.deepfake_detector import load_model, detect_deepfake, detect_deepfake_multi_frame
from Facelog.antispoofing import AntiSpoofingDetector, combined_spoof_detection
from database import AttendanceDatabase

# Initialize InsightFace app with GPU acceleration
print("🚀 Initializing InsightFace with GPU acceleration...")
app = FaceAnalysis(
    name='buffalo_l', 
    root=insightface_model_dir, 
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 for GPU, larger detection size

# Deepfake model (already GPU-enabled)
deepfake_model_path = "Facelog/models/best_deepfake_detector.pth"
deepfake_model = load_model(deepfake_model_path)

# Initialize anti-spoofing detector
print("🛡️ Initializing anti-spoofing detector...")
antispoof_detector = AntiSpoofingDetector()

# Initialize database
db = AttendanceDatabase()
excel_path = "attendance/attendance.xlsx"

# Migrate from Excel if exists and database is empty
if os.path.exists(excel_path):
    # Check if database has students
    student_data = db.get_all_students()
    if not student_data:
        print("📊 Migrating data from Excel to SQLite...")
        db.migrate_from_excel(excel_path)
        student_data = db.get_all_students()
    else:
        print("📊 Using existing SQLite database")
else:
    print("[❌] attendance.xlsx not found!")
    student_data = {}

# Export today's attendance to Excel for compatibility
print("📤 Exporting today's attendance to Excel...")
db.export_to_excel(excel_path)

def monitor_gpu_memory():
    """Monitor GPU memory usage for optimization"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        return f"GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached"
    return "CPU mode"

def combined_multi_frame_detection(face_images, deepfake_model, antispoof_detector, num_frames=7):
    """Combined deepfake and anti-spoofing detection across multiple frames"""
    if len(face_images) == 0:
        return "No images provided", 0, {}
    
    all_results = []
    deepfake_real_scores = []
    antispoof_live_scores = []
    
    print(f"\n🔍 Analyzing {min(len(face_images), num_frames)} frames...")
    
    for i, face_img in enumerate(face_images[:num_frames]):
        print(f"\n--- Frame {i+1} ---")
        result, confidence, details = combined_spoof_detection(face_img, deepfake_model, antispoof_detector)
        if result != "UNKNOWN":
            all_results.append((result, confidence, details))
            # Collect individual scores for averaging
            deepfake_real_scores.append(details.get('deepfake_score', 0) / 100.0)  # Convert to 0-1
            antispoof_live_scores.append(details.get('antispoof_score', 0) / 100.0)  # Convert to 0-1
    
    if len(all_results) == 0:
        return "No valid results", 0, {}
    
    # Calculate averages using exact Step B formula
    avg_deepfake_real = sum(deepfake_real_scores) / len(deepfake_real_scores)
    avg_antispoof_live = sum(antispoof_live_scores) / len(antispoof_live_scores)
    
    # EXACT FORMULA: Combined = (deepfake_real_prob + antispoof_live_prob) / 2
    final_combined_score = (avg_deepfake_real + avg_antispoof_live) / 2
    
    print(f"\n🎯 FINAL MULTI-FRAME CALCULATION:")
    print(f"   Average Deepfake Real Score: {avg_deepfake_real:.3f}")
    print(f"   Average Anti-spoof Live Score: {avg_antispoof_live:.3f}")
    print(f"   Combined = ({avg_deepfake_real:.3f} + {avg_antispoof_live:.3f}) / 2 = {final_combined_score:.3f}")
    
    # Step B threshold check
    if final_combined_score >= 0.70:
        final_prediction = "REAL"
        print(f"✅ FINAL DECISION: ACCEPT (Combined {final_combined_score:.3f} ≥ 0.70)")
    else:
        final_prediction = "FAKE" 
        print(f"❌ FINAL DECISION: REJECT (Combined {final_combined_score:.3f} < 0.70)")
    
    # Count votes for display
    real_count = sum(1 for r, c, d in all_results if r == "REAL")
    fake_count = len(all_results) - real_count
    
    # Aggregate details with all scoring information
    summary_details = {
        'total_frames': len(all_results),
        'real_votes': real_count,
        'fake_votes': fake_count,
        'avg_confidence': final_combined_score * 100,  # Convert to percentage
        'deepfake_avg_score': avg_deepfake_real * 100,
        'antispoof_avg_score': avg_antispoof_live * 100,
        'combined_score': final_combined_score * 100,
        'frames_used': len(all_results),
        'decision': final_prediction
    }
    
    print(f"📊 Votes: {real_count} real, {fake_count} fake across {len(all_results)} frames")
    return final_prediction, round(final_combined_score * 100, 2), summary_details

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
marked_students = db.get_todays_attendance()  # Get from SQLite
print(f"📋 Students already marked today: {len(marked_students)}")

# Frame collection for multi-frame deepfake detection
face_frame_collections = {}  # {student_name: [face_images]}
frame_collection_threshold = 7  # Number of frames to collect

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Periodic GPU memory cleanup for optimal performance
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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

        # Multi-frame deepfake check
        if detected_name in student_data and detected_name not in marked_students:
            x1, y1, x2, y2 = map(int, face.bbox)
            face_img = frame[y1:y2, x1:x2]

            if face_img is not None and face_img.size != 0:
                # Initialize collection for new student
                if detected_name not in face_frame_collections:
                    face_frame_collections[detected_name] = []
                    # Reset liveness tracking for new student
                    antispoof_detector.reset_liveness_tracking()
                    print(f"🆕 Starting liveness detection for {detected_name}")
                
                # Add face image to collection
                face_frame_collections[detected_name].append(face_img.copy())
                
                print(f"📸 Collecting frame {len(face_frame_collections[detected_name])}/{frame_collection_threshold} for {detected_name}")
                
                # When we have enough frames, perform combined multi-frame analysis
                if len(face_frame_collections[detected_name]) >= frame_collection_threshold:
                    result, confidence, details = combined_multi_frame_detection(
                        face_frame_collections[detected_name], 
                        deepfake_model,
                        antispoof_detector,
                        frame_collection_threshold
                    )
                    
                    print(f"🔍 Combined Detection for {detected_name}: {result} ({confidence}%)")
                    print(f"   📊 Analysis: {details.get('real_votes', 0)} real votes, {details.get('fake_votes', 0)} fake votes")
                    print(f"   🚫 Liveness Check: Motion + Blink analysis across {details.get('frames_used', 0)} frames")
                    
                    if result == "REAL" and confidence > 70:  # Higher threshold for combined detection
                        # Mark in SQLite database with detailed scores
                        success = db.mark_attendance(
                            detected_name, 
                            confidence, 
                            'combined_multi_frame',
                            deepfake_avg_score=details.get('deepfake_avg_score'),
                            antispoof_avg_score=details.get('antispoof_avg_score'),
                            combined_score=details.get('combined_score'),
                            frames_used=details.get('frames_used'),
                            decision=details.get('decision')
                        )
                        if success:
                            marked_students.add(detected_name)
                            print(f"✅ Marked {detected_name} as Present (Combined confidence: {confidence}%)")
                            print(f"📊 Detailed scores saved: DF={details.get('deepfake_avg_score', 0):.1f}%, AS={details.get('antispoof_avg_score', 0):.1f}%")
                            
                            # Export updated data to Excel
                            db.export_to_excel(excel_path)
                            print(f"📤 Updated Excel report")
                            
                            # Clear the collection
                            del face_frame_collections[detected_name]
                        else:
                            print(f"❌ Failed to mark attendance for {detected_name}")
                    elif len(face_frame_collections[detected_name]) >= frame_collection_threshold * 2:
                        # If we've collected too many frames without success, reset
                        print(f"⚠️ {detected_name} flagged as {result} ({confidence}%) - Failed liveness checks")
                        print(f"   🚫 Possible replay attack detected - insufficient motion/blink patterns")
                        antispoof_detector.reset_liveness_tracking()
                        del face_frame_collections[detected_name]
            else:
                print(f"[❌] Face image for {detected_name} is empty. Skipping deepfake detection.")

        # Draw box and name
        (x1, y1, x2, y2) = map(int, face.bbox)
        
        # Color coding: Green for known, Red for unknown, Blue for collecting frames
        box_color = (0, 255, 0) if detected_name != "Unknown" else (0, 0, 255)
        if detected_name in face_frame_collections:
            box_color = (255, 165, 0)  # Orange while collecting frames
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        
        # Show frame collection progress
        display_text = detected_name
        if detected_name in face_frame_collections:
            frame_count = len(face_frame_collections[detected_name])
            display_text = f"{detected_name} ({frame_count}/{frame_collection_threshold})"
        
        cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("FaceLog - GPU Powered", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
