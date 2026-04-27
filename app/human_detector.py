import os
import gdown
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms

MODEL_PATH = "models/ai_image_detector_model_improved_convnext.pth"

def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)

        url = "https://drive.google.com/uc?id=1fDPc8Ff3pkhimQMWz5lyZ21EFCSf4HxN"
        gdown.download(url, MODEL_PATH, quiet=False)


class HumanDetector:
    """
    Improved human-focused AI detector.
    This model is trained on the harder human_faces dataset.
    """

    def __init__(self, model_path: str, class_names=None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names or ["fake", "real"]

        # Improved model architecture
        self.model = timm.create_model("convnext_tiny", pretrained=False, num_classes=2)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
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
