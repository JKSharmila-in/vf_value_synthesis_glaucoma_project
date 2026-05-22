import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

from huggingface_hub import hf_hub_download


# -----------------------------
# CONFIG
# -----------------------------
HF_REPO_ID = "JKSharmila/VF_values"
HF_MODEL_FILE = "pytorch_model.bin"   # change if your repo uses safetensors
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# MODEL DEFINITION
# ⚠️ MUST MATCH TRAINING ARCHITECTURE
# -----------------------------
class VFModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 🔴 Example architecture
        # Replace ONLY if your original model differs
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 128),  # adjust if input size differs
            nn.ReLU(),
            nn.Linear(128, 54)  # 👈 number of VF values
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x


# -----------------------------
# LOAD MODEL FROM HUGGING FACE
# -----------------------------
def load_model():
    model = VFModel().to(DEVICE)

    print("Downloading model weights from Hugging Face...")
    ckpt_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILE
    )

    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    print("Model loaded successfully.")
    return model


# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # batch dimension
    return image.to(DEVICE)


# -----------------------------
# MAIN PREDICTION FUNCTION
# -----------------------------
def predict_vf(image_path):
    model = load_model()
    image_tensor = preprocess_image(image_path)

    with torch.no_grad():
        output = model(image_tensor)

    vf_values = output.squeeze().cpu().numpy()

    return vf_values.tolist()  # frontend / API friendly


# -----------------------------
# CLI TEST (OPTIONAL)
# -----------------------------
if __name__ == "__main__":
    test_image = "sample_fundus.jpg"  # change for testing
    vf = predict_vf(test_image)
    print("Predicted VF values:")
    print(vf)
