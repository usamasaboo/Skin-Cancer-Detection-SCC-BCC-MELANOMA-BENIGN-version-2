from flask import Flask, render_template, request
import torch
import timm
from torchvision import transforms
from PIL import Image
import os

app = Flask(__name__)

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "saved_models/efficientnet_b4_skin_cancer_4class.pth"
IMAGE_SIZE = 384
CLASSES = ["bcc", "scc", "melanoma", "benign"]

# ===============================
# LOAD MODEL (ONLY ONCE)
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model(
    "efficientnet_b4",
    pretrained=False,
    num_classes=len(CLASSES)
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print("✅ Model loaded successfully")

# ===============================
# IMAGE TRANSFORM
# ===============================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===============================
# ROUTES
# ===============================
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded"

        file = request.files["file"]

        if file.filename == "":
            return "No selected file"

        try:
            image = Image.open(file).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)

            result = CLASSES[predicted.item()]

            return f"Prediction: {result}"

        except Exception as e:
            return f"Error: {str(e)}"

    return render_template("index.html")

# ===============================
# RUN SERVER (LOCAL ONLY)
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
