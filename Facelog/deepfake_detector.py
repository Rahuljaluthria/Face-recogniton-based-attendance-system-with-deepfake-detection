import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn as nn

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")

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

        checkpoint = torch.load(model_path, map_location=device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(device)  # Move model to GPU
        model.eval()

        print("✅ ResNet-50 model with custom FC layers loaded successfully on", device)
        return model

    except Exception as e:
        print(f"Failed to load ResNet-50 model: {e}")
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

    with torch.no_grad():
        output = model(image)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence = probs[0][1].item()
        prediction = "FAKE" if torch.argmax(output).item() == 1 else "REAL"
        return prediction, round(confidence * 100, 2)


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
