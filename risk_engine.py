class RiskEngine:
    """
    Risk scoring engine.

    Score guide:
    1 = low risk
    2 = mild concern
    3 = moderate risk
    4 = high risk
    5 = severe risk
    """

    def __init__(self):
        self.severe_keywords = {
            "gun", "pistol", "rifle", "firearm",
            "knife", "blade", "weapon",
            "explosion", "grenade", "missile"
        }

        self.conflict_keywords = {
            "fire", "flames", "smoke", "burning",
            "destroyed building", "soldier", "military",
            "military vehicle", "tank", "helicopter"
        }

        self.context_keywords = {
            "person", "crowd", "car", "truck", "bus",
            "train", "airplane", "backpack"
        }

    def _normalise_prediction(self, prediction):
        prediction = str(prediction).lower().strip()

        if prediction in {"fake", "ai_generated", "ai-generated", "ai generated"}:
            return "fake"

        if prediction in {"real", "authentic"}:
            return "real"

        return prediction

    def _label_matches(self, label, keywords):
        label = str(label).lower().strip()
        return any(keyword in label for keyword in keywords)

    def _get_matches(self, detected_objects, keywords, min_conf=0.25):
        matches = []

        for obj in detected_objects:
            label = str(obj.get("label", "")).lower().strip()
            confidence = float(obj.get("confidence", 0.0))

            if confidence < min_conf:
                continue

            if self._label_matches(label, keywords):
                matches.append({
                    "label": label,
                    "confidence": confidence
                })

        return matches

    def _format_matches(self, matches):
        if not matches:
            return ""

        best = {}

        for item in matches:
            label = item["label"]
            confidence = item["confidence"]

            if label not in best or confidence > best[label]:
                best[label] = confidence

        return ", ".join(
            f"{label} ({confidence:.2f})"
            for label, confidence in sorted(best.items())
        )

    def calculate_risk(
        self,
        image,
        final_prediction,
        final_confidence,
        detected_objects,
        general_fake_score,
        human_fake_score
    ):
        final_prediction = self._normalise_prediction(final_prediction)

        general_fake_score = float(general_fake_score)
        human_fake_score = float(human_fake_score)
        final_confidence = float(final_confidence)

        is_fake = final_prediction == "fake"

        severe_hits = self._get_matches(
            detected_objects,
            self.severe_keywords,
            min_conf=0.25
        )

        conflict_hits = self._get_matches(
            detected_objects,
            self.conflict_keywords,
            min_conf=0.25
        )

        context_hits = self._get_matches(
            detected_objects,
            self.context_keywords,
            min_conf=0.30
        )

        contains_person = any(
            "person" in str(obj.get("label", "")).lower()
            and float(obj.get("confidence", 0.0)) >= 0.25
            for obj in detected_objects
        )

        detector_gap = abs(general_fake_score - human_fake_score)

        score = 1
        reasons = []

        # -----------------------------
        # Object-based risk
        # -----------------------------
        if context_hits:
            score = max(score, 2)
            reasons.append(
                f"Context-sensitive objects detected: {self._format_matches(context_hits)}."
            )

        if conflict_hits:
            score = max(score, 4)
            reasons.append(
                f"Conflict or disaster indicators detected: {self._format_matches(conflict_hits)}."
            )

        if severe_hits:
            score = max(score, 5)
            reasons.append(
                f"Severe-risk objects detected: {self._format_matches(severe_hits)}."
            )

        # -----------------------------
        # AI-generation risk
        # -----------------------------
        if is_fake:
            score = max(score, 2)
            reasons.append(
                f"The combined AI detector predicts AI-generated content with confidence {final_confidence:.2f}."
            )

        if general_fake_score >= 0.85:
            score = max(score, 3)
            reasons.append(
                f"The general AI detector strongly indicates AI generation: {general_fake_score:.2f}."
            )

        if human_fake_score >= 0.85:
            score = max(score, 3)
            reasons.append(
                f"The human-focused AI detector strongly indicates AI generation: {human_fake_score:.2f}."
            )

        # -----------------------------
        # Disagreement handling
        # -----------------------------
        if detector_gap >= 0.60:
            score = max(score, 4)
            reasons.append(
                f"The AI detectors strongly disagree "
                f"(general={general_fake_score:.2f}, human={human_fake_score:.2f}), "
                f"so the image needs manual review."
            )

        elif detector_gap >= 0.35:
            score = max(score, 3)
            reasons.append(
                f"The AI detectors partially disagree "
                f"(general={general_fake_score:.2f}, human={human_fake_score:.2f})."
            )

        # -----------------------------
        # Combined risk escalation
        # -----------------------------
        if contains_person and is_fake:
            score = max(score, 3)
            reasons.append(
                "The image appears AI-generated and contains a person, increasing impersonation or misinformation risk."
            )

        if is_fake and conflict_hits:
            score = max(score, 5)
            reasons.append(
                "AI-generation signals are combined with conflict/disaster content."
            )

        if is_fake and severe_hits:
            score = 5
            reasons.append(
                "AI-generation signals are combined with weapons or severe-risk content."
            )

        if contains_person and severe_hits:
            score = 5
            reasons.append(
                "People appear together with severe-risk objects."
            )

        if conflict_hits and severe_hits:
            score = 5
            reasons.append(
                "The scene contains both conflict indicators and severe-risk objects."
            )

        if not reasons:
            reasons.append(
                "No strong AI-generation, weapon, conflict, or high-risk context indicators were detected."
            )

        explanation = "\n".join(f"- {reason}" for reason in reasons)

        return score, explanation