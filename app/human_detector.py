import os
import gdown
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms


MODEL_PATH = "models/ai_image_detector_model_improved_convnext.pth"
MODEL_URL = "https://drive.google.com/uc?id=1fDPc8Ff3pkhimQMWz5lyZ21EFCSf4HxN"


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading human detector model...")
        os.makedirs("models", exist_ok=True)
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model download failed: {MODEL_PATH}")


download_model()


class HumanDetector:
    """
    Improved human-focused AI detector using ConvNeXt-Tiny.
    """

    def __init__(self, model_path: str = MODEL_PATH, class_names=None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names or ["fake", "real"]

        self.model = timm.create_model(
            "convnext_tiny",
            pretrained=False,
            num_classes=2
        )

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def predict(self, image: Image.Image):
        image = image.convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = F.softmax(outputs, dim=1)[0].cpu().numpy()

        pred_idx = int(probs.argmax())

        predicted_class = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])

        probabilities = {
            self.class_names[0]: float(probs[0]),
            self.class_names[1]: float(probs[1])
        }

        return predicted_class, confidence, probabilities
