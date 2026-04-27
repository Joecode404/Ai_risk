from ultralytics import YOLO
from PIL import Image
import numpy as np


class ObjectDetector:
    """
    Uses YOLO to detect visible objects in an image.
    """

    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect_objects(self, image: Image.Image, conf_threshold: float = 0.25):
        image_array = np.array(image.convert("RGB"))
        results = self.model(image_array, conf=conf_threshold)

        detected_objects = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                label = result.names[class_id]

                detected_objects.append({
                    "label": label,
                    "confidence": confidence
                })

        return detected_objects