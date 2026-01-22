#!/usr/bin/env python3
"""
Test script to verify all components are working correctly
"""

import sys
import os
import numpy as np
import cv2

def test_database():
    """Test database functionality"""
    print("🧪 Testing database...")
    try:
        from database import AttendanceDatabase
        db = AttendanceDatabase()
        students = db.get_all_students()
        print(f"✅ Database: {len(students)} students loaded")
        return True
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def test_deepfake_detector():
    """Test deepfake detector"""
    print("🧪 Testing deepfake detector...")
    try:
        from Facelog.deepfake_detector import load_model, detect_deepfake_from_array
        model_path = "Facelog/models/best_deepfake_detector.pth"
        model = load_model(model_path)
        
        if model is not None:
            # Test with dummy image
            dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            result, confidence = detect_deepfake_from_array(dummy_image, model)
            print(f"✅ Deepfake detector: {result} ({confidence}%)")
            return True
        else:
            print("❌ Deepfake model failed to load")
            return False
    except Exception as e:
        print(f"❌ Deepfake detector error: {e}")
        return False

def test_anti_spoofing():
    """Test anti-spoofing detector"""
    print("🧪 Testing anti-spoofing detector...")
    try:
        from Facelog.antispoofing import AntiSpoofingDetector
        detector = AntiSpoofingDetector()
        
        # Test with dummy image
        dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result, confidence = detector.detect_spoofing(dummy_image)
        print(f"✅ Anti-spoofing detector: {result} ({confidence}%)")
        return True
    except Exception as e:
        print(f"❌ Anti-spoofing error: {e}")
        return False

def test_combined_detection():
    """Test combined detection"""
    print("🧪 Testing combined detection...")
    try:
        from Facelog.antispoofing import AntiSpoofingDetector, combined_spoof_detection
        from Facelog.deepfake_detector import load_model
        
        # Load models
        deepfake_model = load_model("Facelog/models/best_deepfake_detector.pth")
        antispoof_detector = AntiSpoofingDetector()
        
        # Test with dummy image
        dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result, confidence, details = combined_spoof_detection(dummy_image, deepfake_model, antispoof_detector)
        
        print(f"✅ Combined detection: {result} ({confidence}%)")
        print(f"   📊 Details: {details}")
        return True
    except Exception as e:
        print(f"❌ Combined detection error: {e}")
        return False

def test_insightface():
    """Test InsightFace"""
    print("🧪 Testing InsightFace...")
    try:
        import os
        os.environ["INSIGHTFACE_HOME"] = "D:/Facerecognitonbasedattendancesystem/insightface_models"
        
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=-1)  # Use CPU for testing
        
        # Test with dummy image
        dummy_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        faces = app.get(dummy_image)
        
        print(f"✅ InsightFace: Detected {len(faces)} faces")
        return True
    except Exception as e:
        print(f"❌ InsightFace error: {e}")
        return False

def main():
    print("🚀 Face Recognition Attendance System - Component Test")
    print("=" * 60)
    
    tests = [
        ("Database", test_database),
        ("Deepfake Detector", test_deepfake_detector),
        ("Anti-Spoofing", test_anti_spoofing),
        ("Combined Detection", test_combined_detection),
        ("InsightFace", test_insightface)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}...")
        results[test_name] = test_func()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\n🏆 Overall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All systems are working correctly!")
        print("\n💡 You can now run: python main.py")
    else:
        print("⚠️ Some components need attention before full deployment")

if __name__ == "__main__":
    main()