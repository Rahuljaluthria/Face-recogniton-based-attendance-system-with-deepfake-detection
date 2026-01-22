import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn as nn

# Set device (GPU if available) with optimizations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print(f"🚀 GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    # Optimize GPU memory usage
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster training
else:
    print(f"💻 Using CPU for processing")

print(f"🔧 PyTorch device: {device}")

def load_model(model_path):
    try:
        model = models.resnet50(weights=None)

        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

        # Load checkpoint with proper device handling
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path, map_location=device)
            print(f"🚀 Loading model on GPU: {torch.cuda.get_device_name(0)}")
        else:
            checkpoint = torch.load(model_path, map_location='cpu')
            print("💻 Loading model on CPU")

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(device)  # Move model to GPU/CPU
        model.eval()  # Set to inference mode
        
        # Check for FP16 capability but don't convert model to half
        if torch.cuda.is_available() and hasattr(torch.cuda, 'get_device_capability'):
            capability = torch.cuda.get_device_capability(0)
            if capability[0] >= 7:  # Tensor cores available (RTX series)
                print("⚡ GPU supports FP16 acceleration via autocast")
            else:
                print("💡 GPU does not support Tensor Cores")

        print(f"✅ ResNet-50 model loaded successfully on {device}")
        return model

    except Exception as e:
        print(f"❌ Failed to load ResNet-50 model: {e}")
        return None


def detect_deepfake(image_path, model):
    if model is None:
        return "Model not loaded.", 0

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return "Invalid image", 0

    image = transform(image).unsqueeze(0).to(device)  # Move image to GPU

    # Ensure model is in eval mode
    model.eval()
    
    with torch.no_grad():
        # Use autocast for automatic mixed precision (FP16)
        if torch.cuda.is_available():
            with torch.amp.autocast('cuda'):
                output = model(image)
        else:
            output = model(image)
            
        probs = torch.nn.functional.softmax(output, dim=1)
        prob_fake = probs[0][1].item()  # Class 1 = FAKE
        prob_real = probs[0][0].item()  # Class 0 = REAL
        
        # Print both probabilities for debugging
        print(f"[REAL={prob_real:.3f}, FAKE={prob_fake:.3f}] (Class 0=REAL, Class 1=FAKE)")
        
        prediction = "FAKE" if torch.argmax(output).item() == 1 else "REAL"
        confidence = prob_fake if prediction == "FAKE" else prob_real
        
        return prediction, confidence


def detect_deepfake_from_array(image_array, model):
    """Detect deepfake from numpy array (frame from camera) with GPU optimization"""
    if model is None:
        return "Model not loaded.", 0

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    try:
        # Convert BGR to RGB for PIL
        image_rgb = image_array[:, :, ::-1] 
        image = Image.fromarray(image_rgb)
    except Exception as e:
        print(f"❌ Error processing image array: {e}")
        return "Invalid image", 0

    # Process and move to device
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Ensure model is in eval mode
    model.eval()

    with torch.no_grad():
        # Clear GPU cache for optimal memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Use autocast for automatic mixed precision (FP16)
        if torch.cuda.is_available():
            with torch.amp.autocast('cuda'):
                output = model(image_tensor)
        else:
            output = model(image_tensor)
            
        probs = torch.nn.functional.softmax(output, dim=1)
        prob_fake = probs[0][1].item()  # Class 1 = FAKE
        prob_real = probs[0][0].item()  # Class 0 = REAL
        
        # Print both probabilities for debugging
        print(f"[REAL={prob_real:.3f}, FAKE={prob_fake:.3f}] (Class 0=REAL, Class 1=FAKE)")
        
        prediction = "FAKE" if torch.argmax(output).item() == 1 else "REAL"
        # Return probability as 0-1 range for consistent math
        confidence = prob_real if prediction == "REAL" else prob_fake
        
        return prediction, confidence


def detect_deepfake_multi_frame(face_images, model, num_frames=7):
    """Detect deepfake across multiple frames and return averaged confidence"""
    if model is None:
        return "Model not loaded.", 0
    
    if len(face_images) == 0:
        return "No images provided", 0
    
    confidences = []
    predictions = []
    
    for face_img in face_images[:num_frames]:  # Use up to num_frames
        prediction, confidence = detect_deepfake_from_array(face_img, model)
        if prediction != "Invalid image":
            confidences.append(confidence)
            predictions.append(1 if prediction == "FAKE" else 0)
    
    if len(confidences) == 0:
        return "No valid images", 0
    
    # Average confidence and get majority vote
    avg_confidence = sum(confidences) / len(confidences)
    avg_prediction = sum(predictions) / len(predictions)
    
    final_prediction = "FAKE" if avg_prediction > 0.5 else "REAL"
    
    print(f"📊 Multi-frame analysis: {len(confidences)} frames, avg confidence: {avg_confidence:.2f}")
    return final_prediction, round(avg_confidence * 100, 2)


# Dummy attendance marking functions
def mark_attendance(student_name):
    print(f"✅ Attendance marked for {student_name}.")


def flag_suspicious(student_name, confidence):
    print(f"⚠️ {student_name}'s image flagged as suspicious ({confidence}% confidence). Manual review needed.")


if __name__ == "__main__":
    model_path = r"/Facelog/models/best_deepfake_detector.pth"
    image_path = r"/Facelog/test_images/Navraj Singh.jpg"
    student_name = "Navraj Singh"

    model = load_model(model_path)
    result, confidence = detect_deepfake(image_path, model)
    print(f"🔍 Prediction: {result} ({confidence}% confidence)")

    if result == "REAL" and confidence < 90:
        print("⚠️ Confidence too low. Manual check advised.")
        flag_suspicious(student_name, confidence)
    elif result == "REAL" and confidence >= 90:
        mark_attendance(student_name)
    else:
        flag_suspicious(student_name, confidence)
