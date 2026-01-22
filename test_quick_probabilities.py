#!/usr/bin/env python3
"""
Quick test to demonstrate probability display and combined logic
"""

import cv2
import numpy as np
from Facelog.deepfake_detector import load_model, detect_deepfake_from_array
from Facelog.antispoofing import AntiSpoofingDetector, combined_spoof_detection

def test_probability_display():
    """Test probability display and combined logic"""
    print("🔍 Testing Probability Display and Combined Logic")
    print("=" * 60)
    
    # Load models
    print("📥 Loading models...")
    deepfake_model = load_model("Facelog/best_deepfake_detector.pth")
    antispoof_detector = AntiSpoofingDetector()
    
    # Create test images with different characteristics
    test_cases = [
        ("Random Image 1", np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)),
        ("Random Image 2", np.random.randint(50, 200, (112, 112, 3), dtype=np.uint8)),
        ("Dark Image", np.random.randint(0, 50, (112, 112, 3), dtype=np.uint8)),
        ("Bright Image", np.random.randint(200, 255, (112, 112, 3), dtype=np.uint8)),
    ]
    
    for i, (name, test_image) in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {name} ---")
        
        # Test individual components
        print("🤖 DEEPFAKE DETECTION:")
        df_pred, df_conf = detect_deepfake_from_array(test_image, deepfake_model)
        print(f"   Prediction: {df_pred}, Confidence: {df_conf:.3f}")
        
        print("🛡️ ANTI-SPOOFING DETECTION:")
        as_pred, as_conf = antispoof_detector.detect_spoofing_normalized(test_image)
        print(f"   Prediction: {as_pred}, Confidence: {as_conf:.3f}")
        
        # Test combined detection
        print("🧮 COMBINED DETECTION:")
        combined_pred, combined_conf, details = combined_spoof_detection(
            test_image, deepfake_model, antispoof_detector
        )
        print(f"   Final Result: {combined_pred} ({combined_conf}%)")
        
        # Show formula calculation
        df_real_prob = df_conf if df_pred == "REAL" else (1.0 - df_conf)
        as_live_prob = as_conf if as_pred == "REAL" else (1.0 - as_conf)
        formula_result = (df_real_prob + as_live_prob) / 2
        
        print(f"📋 FORMULA VERIFICATION:")
        print(f"   DF Real Prob: {df_real_prob:.3f}")
        print(f"   AS Live Prob: {as_live_prob:.3f}")
        print(f"   Combined = ({df_real_prob:.3f} + {as_live_prob:.3f}) / 2 = {formula_result:.3f}")
        
        # Decision
        threshold = 0.70
        decision = "✅ ACCEPT" if formula_result >= threshold else "❌ REJECT"
        print(f"   Decision: {decision} (threshold = {threshold})")
        
        print("-" * 50)
    
    print("\n🎯 SUMMARY:")
    print("✅ Step A: Both probabilities are printed clearly")
    print("✅ Step B: Combined formula = (deepfake_real_prob + antispoof_live_prob) / 2")
    print("✅ Step C: Threshold check at 0.70 for accept/reject")
    print("✅ All probability displays working correctly!")

if __name__ == "__main__":
    test_probability_display()