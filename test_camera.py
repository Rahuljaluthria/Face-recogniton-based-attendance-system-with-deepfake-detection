import cv2

for i in range(5):  # Checking camera indexes from 0 to 4
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ Camera found at index {i}")
        cap.release()
    else:
        print(f"❌ No camera found at index {i}")
