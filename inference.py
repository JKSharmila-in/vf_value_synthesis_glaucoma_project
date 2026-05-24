# inference.py
import io
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision import transforms
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download

# --------------------------------------------------
# Flask setup
# --------------------------------------------------
app = Flask(__name__)
CORS(app)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# CONFIG — Hugging Face
# --------------------------------------------------
HF_REPO_ID = "JKSharmila/VF_values"
HF_MODEL_FILE = "pytorch_model.bin"   # ⚠️ must match HF repo

# --------------------------------------------------
# Model definition (MUST match training)
# --------------------------------------------------
class VFModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(16, 54)   # 54 VF values

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# --------------------------------------------------
# Load model ONCE
# --------------------------------------------------
print("📥 Downloading model from Hugging Face...")
model_path = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename=HF_MODEL_FILE
)

model = VFModel().to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()

print("✅ Model loaded successfully")

# --------------------------------------------------
# Image preprocessing (MUST match training)
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# Prediction API
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        # ---- Read image ----
        image_file = request.files["image"]
        image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # ---- Inference ----
        with torch.no_grad():
            vf_pred = model(image_tensor).cpu().numpy()[0]

        vf_values = vf_pred.round(2).tolist()

        # ---- Derived metrics ----
        md = round(float(np.mean(vf_values)), 2)
        psd = round(float(np.std(vf_values)), 2)

        return jsonify({
            "vf_values": vf_values,
            "md": md,
            "psd": psd
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------------------------------------
# Run server
# --------------------------------------------------
if __name__ == "__main__":
    print("🚀 Backend running at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
