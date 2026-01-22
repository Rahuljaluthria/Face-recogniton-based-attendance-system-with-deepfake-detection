#!/usr/bin/env python3
"""
Test replay protection features (motion and blink detection)
"""

import numpy as np
import cv2
from Facelog.antispoofing import AntiSpoofingDetector

def simulate_static_image_attack():
    """Simulate a static photo replay attack"""
    print("🎭 Simulating Static Photo Attack...")
    detector = AntiSpoofingDetector()
    
    # Create a static face-like image
    static_image = np.ones((200, 200, 3), dtype=np.uint8) * 128
    # Add some simple facial features (squares for eyes, nose, mouth)
    cv2.rectangle(static_image, (50, 60), (80, 80), (50, 50, 50), -1)  # Left eye
    cv2.rectangle(static_image, (120, 60), (150, 80), (50, 50, 50), -1)  # Right eye
    cv2.rectangle(static_image, (90, 90), (110, 120), (100, 100, 100), -1)  # Nose
    cv2.rectangle(static_image, (70, 140), (130, 160), (80, 80, 80), -1)  # Mouth
    
    results = []
    print("   Analyzing static image across 10 frames...")
    
    for frame_num in range(10):
        result, confidence = detector.detect_spoofing(static_image)
        results.append((frame_num + 1, result, confidence))
        print(f"   Frame {frame_num + 1}: {result} ({confidence:.1f}%)")
    
    # Calculate detection accuracy
    fake_detections = sum(1 for _, result, _ in results if result == "FAKE")
    print(f"\n📊 Static Image Attack Results:")
    print(f"   Frames analyzed: {len(results)}")
    print(f"   Detected as FAKE: {fake_detections}/{len(results)} ({fake_detections/len(results)*100:.1f}%)")
    print(f"   Average confidence: {sum(conf for _, _, conf in results)/len(results):.1f}%")
    
    return fake_detections >= len(results) * 0.6  # Consider success if >60% detected as fake

def simulate_motion_attack():
    """Simulate a moving photo/video replay attack"""
    print("\n🎭 Simulating Moving Photo Attack...")
    detector = AntiSpoofingDetector()
    
    results = []
    print("   Analyzing slightly moving image across 10 frames...")
    
    for frame_num in range(10):
        # Create slightly different images to simulate minimal movement
        base_image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        
        # Add slight offset to simulate minimal movement
        offset_x = (frame_num % 3) - 1  # -1, 0, 1 pixel movement
        offset_y = (frame_num % 3) - 1
        
        # Add facial features with slight movement
        cv2.rectangle(base_image, (50 + offset_x, 60 + offset_y), (80 + offset_x, 80 + offset_y), (50, 50, 50), -1)
        cv2.rectangle(base_image, (120 + offset_x, 60 + offset_y), (150 + offset_x, 80 + offset_y), (50, 50, 50), -1)
        cv2.rectangle(base_image, (90 + offset_x, 90 + offset_y), (110 + offset_x, 120 + offset_y), (100, 100, 100), -1)
        cv2.rectangle(base_image, (70 + offset_x, 140 + offset_y), (130 + offset_x, 160 + offset_y), (80, 80, 80), -1)
        
        result, confidence = detector.detect_spoofing(base_image)
        results.append((frame_num + 1, result, confidence))
        print(f"   Frame {frame_num + 1}: {result} ({confidence:.1f}%)")
    
    # Calculate detection accuracy
    real_detections = sum(1 for _, result, _ in results if result == "REAL")
    print(f"\n📊 Moving Photo Attack Results:")
    print(f"   Frames analyzed: {len(results)}")
    print(f"   Detected as REAL: {real_detections}/{len(results)} ({real_detections/len(results)*100:.1f}%)")
    print(f"   Average confidence: {sum(conf for _, _, conf in results)/len(results):.1f}%")
    
    return real_detections >= len(results) * 0.7  # Consider success if >70% detected as real

def simulate_natural_movement():
    """Simulate natural human movement patterns"""
    print("\n🙋‍♂️ Simulating Natural Human Movement...")
    detector = AntiSpoofingDetector()
    
    results = []
    print("   Analyzing natural movement patterns across 10 frames...")
    
    for frame_num in range(10):
        # Create more natural movement patterns
        base_image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        
        # Natural head movement (more random)
        offset_x = int(5 * np.sin(frame_num * 0.3)) + np.random.randint(-2, 3)
        offset_y = int(3 * np.cos(frame_num * 0.4)) + np.random.randint(-1, 2)
        
        # Simulate natural color variations
        brightness_variation = int(20 * np.sin(frame_num * 0.5))
        base_color = 128 + brightness_variation
        
        base_image[:, :, :] = base_color
        
        # Add facial features with natural movement and blinking simulation
        # Simulate blinking by varying eye height
        eye_height = 20 if frame_num % 8 != 3 else 5  # Blink on frame 3, 11, 19, etc.
        
        cv2.rectangle(base_image, (50 + offset_x, 60 + offset_y), 
                     (80 + offset_x, 60 + offset_y + eye_height), (50, 50, 50), -1)
        cv2.rectangle(base_image, (120 + offset_x, 60 + offset_y), 
                     (150 + offset_x, 60 + offset_y + eye_height), (50, 50, 50), -1)
        cv2.rectangle(base_image, (90 + offset_x, 90 + offset_y), 
                     (110 + offset_x, 120 + offset_y), (100, 100, 100), -1)
        cv2.rectangle(base_image, (70 + offset_x, 140 + offset_y), 
                     (130 + offset_x, 160 + offset_y), (80, 80, 80), -1)
        
        result, confidence = detector.detect_spoofing(base_image)
        results.append((frame_num + 1, result, confidence))
        print(f"   Frame {frame_num + 1}: {result} ({confidence:.1f}%) {'👁️' if eye_height == 5 else ''}")
    
    # Calculate detection accuracy
    real_detections = sum(1 for _, result, _ in results if result == "REAL")
    print(f"\n📊 Natural Movement Results:")
    print(f"   Frames analyzed: {len(results)}")
    print(f"   Detected as REAL: {real_detections}/{len(results)} ({real_detections/len(results)*100:.1f}%)")
    print(f"   Average confidence: {sum(conf for _, _, conf in results)/len(results):.1f}%")
    
    return real_detections >= len(results) * 0.8  # Consider success if >80% detected as real

def main():
    print("🛡️ REPLAY PROTECTION TESTING")
    print("=" * 60)
    print("Testing motion-based anti-spoofing with:")
    print("• Head movement detection")
    print("• Blink pattern analysis")
    print("• Texture and color analysis")
    print("=" * 60)
    
    tests = [
        ("Static Photo Attack", simulate_static_image_attack),
        ("Moving Photo Attack", simulate_motion_attack), 
        ("Natural Human Movement", simulate_natural_movement)
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    print("\n" + "=" * 60)
    print("🎯 REPLAY PROTECTION TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<25} {status}")
    
    overall_success = all(results.values())
    print(f"\n🏆 Overall Replay Protection: {'✅ EFFECTIVE' if overall_success else '⚠️ NEEDS IMPROVEMENT'}")
    
    if overall_success:
        print("\n🎉 Replay protection is working effectively!")
        print("   • Static images are detected as fake")
        print("   • Natural movement is recognized as real") 
        print("   • Motion and blink analysis provides additional security")
    else:
        print("\n⚠️ Some tests failed. Consider adjusting thresholds.")

if __name__ == "__main__":
    main()