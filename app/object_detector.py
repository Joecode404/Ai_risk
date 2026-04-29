import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


class ObjectDetector:
    """
    Open-set object detector using Grounding DINO.

    It can search for custom objects like:
    gun, knife, weapon, fire, smoke, explosion, soldier, tank, etc.
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

        self.default_queries = [
            "person",
            "crowd",
            "gun",
            "pistol",
            "rifle",
            "firearm",
            "knife",
            "blade",
            "weapon",
            "fire",
            "flames",
            "smoke",
            "explosion",
            "burning building",
            "destroyed building",
            "soldier",
            "military vehicle",
            "tank",
            "helicopter",
            "airplane",
            "car",
            "truck",
            "bus",
            "train",
            "backpack"
        ]

    def detect_objects(
        self,
        image: Image.Image,
        conf_threshold: float = 0.25,
        text_threshold: float = 0.25,
        queries=None
    ):
        image = image.convert("RGB")
        queries = queries or self.default_queries

        # Grounding DINO prefers labels separated by periods.
        text_prompt = ". ".join(queries) + "."

        inputs = self.processor(
            images=image,
            text=text_prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=conf_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]
        )[0]

        detected_objects = []

        boxes = results["boxes"]
        scores = results["scores"]
        labels = results["labels"]

        for box, score, label in zip(boxes, scores, labels):
            label = str(label).lower().strip()
            confidence = float(score.item())

            x1, y1, x2, y2 = box.tolist()

            detected_objects.append({
                "label": label,
                "confidence": confidence,
                "box": [x1, y1, x2, y2]
            })

        detected_objects = self._deduplicate(detected_objects)

        return detected_objects

    def _deduplicate(self, detected_objects):
        """
        Keeps the highest-confidence version of similar labels.
        """
        best = {}

        for obj in detected_objects:
            label = obj["label"]
            confidence = obj["confidence"]

            if label not in best or confidence > best[label]["confidence"]:
                best[label] = obj

        return sorted(
            best.values(),
            key=lambda x: x["confidence"],
            reverse=True
        )