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
HF_MODEL_FILE = "pytorch_model.bin"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_VF_POINTS = 54
INPUT_SIZE = 224


# -----------------------------
# MODEL DEFINITION
# ⚠️ MUST MATCH TRAINING EXACTLY
# -----------------------------
class VFModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, NUM_VF_POINTS)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.fc(x)


# -----------------------------
# LOAD MODEL (ONCE)
# -----------------------------
def load_model():
    model = VFModel().to(DEVICE)

    ckpt_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILE
    )

    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    return model


MODEL = load_model()


# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def preprocess_image(image_path: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = _transform(image).unsqueeze(0)
    return image.to(DEVICE)


# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_vf(image_path: str) -> list:
    image_tensor = preprocess_image(image_path)

    with torch.no_grad():
        output = MODEL(image_tensor)

    vf_values = output.squeeze().cpu().numpy()
    return vf_values.tolist()


# -----------------------------
# CLI TEST
# -----------------------------
if __name__ == "__main__":
    test_image = "sample_fundus.jpg"
    preds = predict_vf(test_image)
    print(f"Predicted {len(preds)} VF values:")
    print(np.round(preds, 2))
