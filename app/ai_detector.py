import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import ConvNextForImageClassification, AutoImageProcessor

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


class AIDetector:
    """
    General AI image detector using your trained ConvNeXt-Tiny model.

    Supports either:
    1. A single .pt checkpoint saved like:
       {
           "model_state_dict": model.state_dict(),
           "config": model.config
       }

    2. A Hugging Face model folder containing:
       config.json
       model.safetensors
       preprocessor_config.json
    """

    def __init__(self, model_path: str, class_names=None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # IMPORTANT:
        # Your training used Label_A where:
        # 0 = REAL
        # 1 = AI_GENERATED / fake
        self.class_names = class_names or ["real", "fake"]

        if model_path.endswith(".pt") or model_path.endswith(".pth"):
            # New single-file PyTorch checkpoint
            # weights_only=False is needed because this checkpoint contains a Hugging Face config object.
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model = ConvNextForImageClassification(checkpoint["config"])
            self.model.load_state_dict(checkpoint["model_state_dict"])

            # Manual preprocessing that matches facebook/convnext-tiny-224 defaults.
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            self.processor = None
        else:
            # Hugging Face saved model folder option
            self.model = ConvNextForImageClassification.from_pretrained(model_path)
            self.processor = AutoImageProcessor.from_pretrained(model_path)
            self.transform = None

        self.model.to(self.device)
        self.model.eval()

    def predict(self, image: Image.Image):
        image = image.convert("RGB")

        if self.processor is not None:
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            inputs = {"pixel_values": image_tensor}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)[0].cpu().numpy()

        pred_idx = int(probs.argmax())
        predicted_class = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])

        probabilities = {
            self.class_names[0]: float(probs[0]),
            self.class_names[1]: float(probs[1])
        }

        return predicted_class, confidence, probabilities
