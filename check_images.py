from PIL import Image

image_path = "faces/Rahul.jpg" # Update with the correct filename if needed

try:
    img = Image.open(image_path)
    img.verify()  # Verify if the image is valid
    print("[✅] Image is valid!")
except Exception as e:
    print(f"[❌] Invalid image: {e}")
