import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


class ObjectDetector:
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

        self.context_queries = [
            "person",
            "face",
            "car",
            "truck",
            "bus",
            "train",
            "airplane",
            "backpack",
            "fire",
            "flames",
            "smoke",
            "explosion",
            "tank",
            "military vehicle",
            "helicopter"
        ]

        self.weapon_queries = [
            "gun",
            "pistol",
            "handgun",
            "rifle",
            "knife",
            "blade"
        ]

    def _run_dino(
        self,
        image: Image.Image,
        queries,
        conf_threshold: float,
        text_threshold: float
    ):
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

        objects = []

        for box, score, label in zip(
            results["boxes"],
            results["scores"],
            results["labels"]
        ):
            x1, y1, x2, y2 = box.tolist()

            objects.append({
                "label": str(label).lower().strip(),
                "confidence": float(score.item()),
                "box": [x1, y1, x2, y2]
            })

        return objects

    def detect_objects(self, image: Image.Image):
        image = image.convert("RGB")

        # Pass 1: strict general/context detection
        context_objects = self._run_dino(
            image=image,
            queries=self.context_queries,
            conf_threshold=0.38,
            text_threshold=0.32
        )

        # Pass 2: more sensitive weapon-only detection
        weapon_objects = self._run_dino(
            image=image,
            queries=self.weapon_queries,
            conf_threshold=0.25,
            text_threshold=0.20
        )

        detected_objects = context_objects + weapon_objects

        return self._deduplicate(detected_objects)

    def _deduplicate(self, detected_objects):
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