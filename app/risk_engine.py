from PIL import Image
import numpy as np


class RiskEngine:
    def __init__(self):
        self.danger_keywords = {
            "knife", "gun", "weapon", "fire", "smoke", "explosion",
            "tank", "helicopter", "soldier", "military", "rifle", "pistol"
        }

        self.context_keywords = {
            "person", "train", "truck", "car", "bus"
        }

    def _visual_fire_smoke_score(self, image: Image.Image):
        """
        Simple image-level heuristic for fire/smoke/war-zone scenes.
        This catches cases where YOLO misses flames, smoke, guns, or knives.
        """
        img = image.convert("RGB").resize((256, 256))
        arr = np.asarray(img).astype(np.float32)

        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]

        # Fire-like pixels: strong red/orange, brighter than blue
        fire_mask = (
            (r > 120) &
            (g > 45) &
            (r > g * 1.15) &
            (g > b * 1.25)
        )

        # Smoke/dark destruction-like pixels
        brightness = (r + g + b) / 3
        smoke_mask = (
            (brightness < 95) &
            (np.abs(r - g) < 35) &
            (np.abs(g - b) < 35)
        )

        fire_ratio = fire_mask.mean()
        smoke_ratio = smoke_mask.mean()

        return fire_ratio, smoke_ratio

    def calculate_risk(
        self,
        image,
        final_prediction,
        final_confidence,
        detected_objects,
        general_fake_score,
        human_fake_score
    ):
        labels = [obj.get("label", "").lower() for obj in detected_objects]

        contains_person = "person" in labels

        danger_hits = [
            label for label in labels
            if any(word in label for word in self.danger_keywords)
        ]

        context_hits = [
            label for label in labels
            if any(word in label for word in self.context_keywords)
        ]

        fire_ratio, smoke_ratio = self._visual_fire_smoke_score(image)

        visual_fire = fire_ratio > 0.015
        visual_smoke = smoke_ratio > 0.25
        visual_warzone = visual_fire and visual_smoke

        is_fake = final_prediction.lower() == "fake"

        score = 1
        reasons = []

        if context_hits:
            score = max(score, 2)
            reasons.append(
                f"Context-sensitive objects detected: {', '.join(sorted(set(context_hits)))}."
            )

        if danger_hits:
            score = max(score, 4)
            reasons.append(
                f"Dangerous objects detected by YOLO: {', '.join(sorted(set(danger_hits)))}."
            )

        if visual_fire:
            score = max(score, 4)
            reasons.append(
                f"Image-level fire/explosion pattern detected."
            )

        if visual_smoke:
            score = max(score, 3)
            reasons.append(
                f"Large dark smoke/destruction pattern detected."
            )

        if visual_warzone:
            score = max(score, 4)
            reasons.append(
                "The overall scene visually resembles a conflict, fire, or war-zone environment."
            )

        if is_fake:
            score = max(score, 2)
            reasons.append(
                f"AI detector predicts FAKE with confidence {final_confidence:.2f}."
            )

        if is_fake and contains_person:
            score = max(score, 3)
            reasons.append(
                "AI-generated image contains people, increasing misinformation risk."
            )

        if is_fake and (danger_hits or visual_warzone):
            score = 5
            reasons.append(
                "AI-generation signals combined with dangerous or conflict-style content."
            )

        if not is_fake and visual_warzone:
            score = max(score, 4)
            reasons.append(
                "Even though the AI detector predicts REAL, the visual content is still high-risk."
            )

        if max(general_fake_score, human_fake_score) > 0.9:
            score = max(score, 3)
            reasons.append(
                "At least one detector gave a very high fake score."
            )

        if not reasons:
            reasons.append("No major risk indicators were detected.")

        explanation = "\n".join(f"- {reason}" for reason in reasons)

        return score, explanation