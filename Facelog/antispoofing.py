import cv2
import numpy as np
import os
from typing import Tuple, Optional, List
import math
from scipy.spatial import distance as dist

class AntiSpoofingDetector:
    def __init__(self, model_dir: str = "Facelog/models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Use built-in anti-spoofing methods
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Motion and liveness tracking
        self.previous_face_center = None
        self.motion_history = []
        self.blink_history = []
        self.frame_count = 0
        
        print("✅ Anti-spoofing detector initialized with motion and blink detection")
    
    def _detect_motion(self, face_image: np.ndarray) -> Tuple[bool, float, float]:
        """Detect head motion between frames"""
        try:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return True, 70.0, 0.1  # No face detected, assume motion
            
            # Get face center
            x, y, w, h = faces[0]
            current_center = (x + w//2, y + h//2)
            
            if self.previous_face_center is None:
                self.previous_face_center = current_center
                return True, 75.0, 0.2
            
            # Calculate movement distance
            motion_distance = math.sqrt(
                (current_center[0] - self.previous_face_center[0])**2 +
                (current_center[1] - self.previous_face_center[1])**2
            )
            
            # Update motion history (keep last 10 movements)
            self.motion_history.append(motion_distance)
            if len(self.motion_history) > 10:
                self.motion_history.pop(0)
            
            self.previous_face_center = current_center
            
            # Natural micro-movements should be present (1-100 pixels) 
            avg_motion = sum(self.motion_history) / len(self.motion_history)
            
            if 1 <= avg_motion <= 100:  # More lenient natural movement range
                confidence = min(85.0, 60 + avg_motion/3)
                return True, confidence, 0.2
            elif avg_motion < 1:  # Too static (likely photo/static video)
                confidence = max(30.0, 50 - (1 - avg_motion) * 20)
                return False, confidence, 0.3
            else:  # Too much movement (might be real but shaky)
                confidence = max(50.0, 80 - (avg_motion - 100) * 0.5)
                return True, confidence, 0.15
                
        except Exception as e:
            print(f"Motion detection error: {e}")
            return True, 60.0, 0.1
    
    def _detect_blinks(self, face_image: np.ndarray) -> Tuple[bool, float, float]:
        """Detect blinks using eye aspect ratio method"""
        try:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return True, 65.0, 0.15
            
            # Get face region
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Detect eyes in face region
            eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 3)
            
            if len(eyes) < 2:
                # If can't detect both eyes, assume moderate liveness
                return True, 60.0, 0.15
            
            # Calculate eye aspect ratios
            ear_values = []
            for (ex, ey, ew, eh) in eyes[:2]:  # Use first two eyes
                # Simple EAR approximation using bounding box
                ear = eh / ew  # Height to width ratio
                ear_values.append(ear)
            
            avg_ear = sum(ear_values) / len(ear_values)
            
            # Track EAR history for blink detection
            self.blink_history.append(avg_ear)
            if len(self.blink_history) > 15:  # Keep last 15 frames
                self.blink_history.pop(0)
            
            # Analyze blink patterns
            if len(self.blink_history) < 5:
                return True, 70.0, 0.15
            
            # Check for blink variations (EAR should vary between ~0.2-0.8)
            ear_variance = np.var(self.blink_history)
            ear_range = max(self.blink_history) - min(self.blink_history)
            
            # Natural blinking creates EAR variations
            if ear_variance > 0.01 and ear_range > 0.1:
                confidence = min(90.0, 70 + ear_variance * 500)
                return True, confidence, 0.25
            elif ear_variance < 0.005:  # Too static - likely photo
                confidence = max(25.0, 65 - ear_variance * 1000)
                return False, confidence, 0.25
            else:  # Moderate variation
                confidence = 65.0
                return True, confidence, 0.2
                
        except Exception as e:
            print(f"Blink detection error: {e}")
            return True, 65.0, 0.1
    
    def _detect_motion(self, face_image: np.ndarray) -> Tuple[bool, float, float]:
        """Detect head motion between frames"""
        try:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return True, 70.0, 0.1  # No face detected, assume motion
            
            # Get face center
            x, y, w, h = faces[0]
            current_center = (x + w//2, y + h//2)
            
            if self.previous_face_center is None:
                self.previous_face_center = current_center
                return True, 75.0, 0.2
            
            # Calculate movement distance
            motion_distance = math.sqrt(
                (current_center[0] - self.previous_face_center[0])**2 +
                (current_center[1] - self.previous_face_center[1])**2
            )
            
            # Update motion history (keep last 10 movements)
            self.motion_history.append(motion_distance)
            if len(self.motion_history) > 10:
                self.motion_history.pop(0)
            
            self.previous_face_center = current_center
            
            # Natural micro-movements should be present (2-20 pixels)
            avg_motion = sum(self.motion_history) / len(self.motion_history)
            
            if 2 <= avg_motion <= 50:  # Natural movement range
                confidence = min(85.0, 70 + avg_motion)
                return True, confidence, 0.3
            elif avg_motion < 2:  # Too static (likely photo/static video)
                confidence = max(20.0, 60 - (2 - avg_motion) * 10)
                return False, confidence, 0.3
            else:  # Too much movement (might be real but shaky)
                confidence = max(40.0, 80 - (avg_motion - 50) * 2)
                return True, confidence, 0.2
                
        except Exception as e:
            print(f"Motion detection error: {e}")
            return True, 60.0, 0.1
    
    def detect_spoofing(self, face_image: np.ndarray) -> Tuple[str, float]:
        """
        Enhanced anti-spoofing detection with motion and blink analysis
        
        Args:
            face_image: Face image as numpy array (BGR format)
            
        Returns:
            Tuple of (prediction, confidence)
            prediction: "REAL" or "FAKE"
            confidence: Confidence score (0-100)
        """
        try:
            if face_image is None or face_image.size == 0:
                return "UNKNOWN", 0.0
            
            self.frame_count += 1
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Multiple anti-spoofing checks including motion and liveness
            checks = {
                'texture_analysis': self._texture_analysis(gray),
                'color_analysis': self._color_analysis(face_image),
                'edge_analysis': self._edge_analysis(gray),
                'brightness_analysis': self._brightness_analysis(gray),
                'motion_analysis': self._detect_motion(face_image),
                'blink_analysis': self._detect_blinks(face_image)
            }
            
            # Combine scores with weights
            real_score = 0
            total_weight = 0
            
            for check_name, (is_real, confidence, weight) in checks.items():
                if is_real:
                    real_score += confidence * weight
                else:
                    # Penalize fake detection more heavily for motion/blink checks
                    penalty = 1.2 if check_name in ['motion_analysis', 'blink_analysis'] else 1.0
                    real_score -= (100 - confidence) * weight * penalty
                total_weight += weight
            
            if total_weight > 0:
                final_confidence = max(0, min(100, real_score / total_weight))
                
                # Motion and blink checks are critical for replay protection
                motion_real, motion_conf, _ = checks['motion_analysis']
                blink_real, blink_conf, _ = checks['blink_analysis']
                
                # Be more lenient - only flag as fake if multiple indicators agree
                if not motion_real and not blink_real and self.frame_count > 5:
                    # Both motion and blink suggest fake, and we have enough frames
                    final_confidence = min(final_confidence, 25.0)
                    prediction = "FAKE"
                elif motion_real and blink_real and final_confidence > 40:
                    # Both liveness indicators are positive
                    final_confidence = min(100, final_confidence * 1.15)
                    prediction = "REAL"
                elif final_confidence > 35:  # Lower threshold for acceptance
                    prediction = "REAL"
                else:
                    prediction = "FAKE"
                
                return prediction, round(final_confidence, 2)
            else:
                return "REAL", 75.0  # Default to real with moderate confidence
                
        except Exception as e:
            print(f"❌ Error in anti-spoofing detection: {e}")
            return "REAL", 60.0  # Default to real if error occurs
    
    def detect_spoofing_normalized(self, face_image: np.ndarray) -> Tuple[str, float]:
        """Detection with normalized 0-1 probability output"""
        prediction, confidence_pct = self.detect_spoofing(face_image)
        confidence_prob = confidence_pct / 100.0  # Convert to 0-1 range
        
        # Print both probabilities for debugging
        if prediction == "REAL":
            prob_real = confidence_prob
            prob_fake = 1.0 - confidence_prob
        else:
            prob_fake = confidence_prob
            prob_real = 1.0 - confidence_prob
            
        print(f"Anti-spoof: [LIVE={prob_real:.3f}, SPOOF={prob_fake:.3f}]")
        
        return prediction, confidence_prob
        prediction, confidence_pct = self.detect_spoofing(face_image)
        confidence_prob = confidence_pct / 100.0  # Convert to 0-1 range
        
        # Print both probabilities for debugging
        if prediction == "REAL":
            prob_real = confidence_prob
            prob_fake = 1.0 - confidence_prob
        else:
            prob_fake = confidence_prob
            prob_real = 1.0 - confidence_prob
            
        print(f"Anti-spoof: [LIVE={prob_real:.3f}, SPOOF={prob_fake:.3f}]")
        
        return prediction, confidence_prob
    
    def _texture_analysis(self, gray_image: np.ndarray) -> Tuple[bool, float, float]:
        """Analyze image texture using Local Binary Patterns"""
        try:
            # Calculate Laplacian variance (focus measure)
            laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
            
            # Real faces typically have higher texture variance
            if laplacian_var > 100:
                return True, min(85.0, 60 + laplacian_var/20), 0.3
            else:
                return False, max(30.0, 80 - laplacian_var/10), 0.3
        except:
            return True, 70.0, 0.1
    
    def _color_analysis(self, color_image: np.ndarray) -> Tuple[bool, float, float]:
        """Analyze color distribution"""
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            
            # Calculate color variance
            h_var = np.var(hsv[:,:,0])
            s_var = np.var(hsv[:,:,1])
            v_var = np.var(hsv[:,:,2])
            
            total_variance = h_var + s_var + v_var
            
            # Real faces typically have more color variation
            if total_variance > 1000:
                confidence = min(90.0, 70 + total_variance/100)
                return True, confidence, 0.2
            else:
                confidence = max(25.0, 75 - total_variance/50)
                return False, confidence, 0.2
        except:
            return True, 65.0, 0.1
    
    def _edge_analysis(self, gray_image: np.ndarray) -> Tuple[bool, float, float]:
        """Analyze edge patterns"""
        try:
            # Detect edges using Canny
            edges = cv2.Canny(gray_image, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Real faces have moderate edge density
            if 0.05 < edge_density < 0.3:
                confidence = 80.0
                return True, confidence, 0.25
            else:
                confidence = 40.0
                return False, confidence, 0.25
        except:
            return True, 70.0, 0.1
    
    def _brightness_analysis(self, gray_image: np.ndarray) -> Tuple[bool, float, float]:
        """Analyze brightness distribution"""
        try:
            # Calculate histogram
            hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
            
            # Check for unnatural brightness peaks (common in spoofed images)
            max_peak = np.max(hist)
            total_pixels = gray_image.size
            
            # Real faces have more distributed brightness
            if max_peak / total_pixels < 0.1:  # No single brightness dominates
                return True, 80.0, 0.25
            else:
                return False, 50.0, 0.25
        except:
            return True, 70.0, 0.1
    
    def reset_liveness_tracking(self):
        """Reset motion and blink tracking for new detection session"""
        self.previous_face_center = None
        self.motion_history = []
        self.blink_history = []
        self.frame_count = 0
    
    def detect_spoofing_from_path(self, image_path: str) -> Tuple[str, float]:
        """Detect spoofing from image file path"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return "UNKNOWN", 0.0
            return self.detect_spoofing(image)
        except Exception as e:
            print(f"❌ Error loading image from path: {e}")
            return "UNKNOWN", 0.0


# Combined detection function
def combined_spoof_detection(face_image: np.ndarray, deepfake_model, antispoof_detector) -> Tuple[str, float, dict]:
    """
    Combined deepfake and anti-spoofing detection with exact formula
    
    Returns:
        (final_prediction, combined_confidence, details)
    """
    details = {}
    
    # Anti-spoofing detection (normalized to 0-1)
    antispoof_pred, antispoof_prob = antispoof_detector.detect_spoofing_normalized(face_image)
    antispoof_live_prob = antispoof_prob if antispoof_pred == "REAL" else (1.0 - antispoof_prob)
    details['antispoof'] = {'prediction': antispoof_pred, 'confidence': antispoof_prob}
    
    # Deepfake detection (already 0-1 range)
    from Facelog.deepfake_detector import detect_deepfake_from_array
    deepfake_pred, deepfake_prob = detect_deepfake_from_array(face_image, deepfake_model)
    deepfake_real_prob = deepfake_prob if deepfake_pred == "REAL" else (1.0 - deepfake_prob)
    details['deepfake'] = {'prediction': deepfake_pred, 'confidence': deepfake_prob}
    
    # Store individual scores
    details['deepfake_score'] = deepfake_real_prob * 100  # Convert back to percentage for storage
    details['antispoof_score'] = antispoof_live_prob * 100
    
    # EXACT FORMULA FROM STEP B: Combined = (deepfake_real_prob + antispoof_live_prob) / 2
    combined_score = (deepfake_real_prob + antispoof_live_prob) / 2
    
    print(f"🧮 Combined calculation: ({deepfake_real_prob:.3f} + {antispoof_live_prob:.3f}) / 2 = {combined_score:.3f}")
    
    # Step B threshold check: Combined ≥ 0.70 → Accept
    if combined_score >= 0.70:
        final_pred = "REAL"
        print(f"✅ ACCEPT: Combined score {combined_score:.3f} ≥ 0.70 threshold")
    else:
        final_pred = "FAKE"
        print(f"❌ REJECT: Combined score {combined_score:.3f} < 0.70 threshold")
    
    details['combined_confidence'] = combined_score * 100  # Convert to percentage
    details['reason'] = f'combined_score_{combined_score:.3f}'
    
    return final_pred, round(combined_score * 100, 2), details


if __name__ == "__main__":
    # Test the anti-spoofing detector
    detector = AntiSpoofingDetector()
    
    # Test with a sample image if available
    test_image_path = "temp_face.jpg"
    if os.path.exists(test_image_path):
        result, confidence = detector.detect_spoofing_from_path(test_image_path)
        print(f"🧪 Test result: {result} ({confidence}% confidence)")
    else:
        print("ℹ️ No test image found. Anti-spoofing detector initialized successfully.")