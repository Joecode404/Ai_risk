class RiskEngine:
    """
    Risk scoring engine.
    Score guide:
    1 = low risk
    2 = AI-generated but low-risk / mild concern
    3 = moderate risk
    4 = high risk
    5 = severe risk
    """

    def __init__(self):
        self.severe_keywords = {
            "gun", "pistol", "handgun", "rifle",
            "knife"
        }

        self.conflict_keywords = {
            "fire", "flames", "smoke", "explosion",
            "tank", "military vehicle", "helicopter"
        }

        self.context_keywords = {
            "person", "face", "crowd"
        }

    def _normalise_prediction(self, prediction):
        prediction = str(prediction).lower().strip()

        if prediction in {"fake", "ai_generated", "ai-generated", "ai generated"}:
            return "fake"

        if prediction in {"real", "authentic"}:
            return "real"

        return prediction

    def _get_matches(self, detected_objects, keywords, min_conf):
        matches = []

        for obj in detected_objects:
            label = str(obj.get("label", "")).lower().strip()
            confidence = float(obj.get("confidence", 0.0))

            if confidence < min_conf:
                continue

            if any(keyword in label for keyword in keywords):
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

    def _filter_severe_hits(self, severe_hits):
        """
        Extra safeguard against false positives.
        Grounding DINO can sometimes mistake sharp shapes,
        such as mountain edges, for blades/weapons.
        This filter only accepts stronger weapon detections.
        """
        filtered = []

        for hit in severe_hits:
            label = hit["label"]
            confidence = hit["confidence"]

            # Guns and knives must be detected with reasonable confidence.
            if label in {"gun", "pistol", "handgun", "rifle", "knife"}:
                if confidence >= 0.50:
                    filtered.append(hit)

        return filtered

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
            min_conf=0.50
        )

        severe_hits = self._filter_severe_hits(severe_hits)

        conflict_hits = self._get_matches(
            detected_objects,
            self.conflict_keywords,
            min_conf=0.50
        )

        context_hits = self._get_matches(
            detected_objects,
            self.context_keywords,
            min_conf=0.40
        )

        contains_person = len(context_hits) > 0

        score = 1
        reasons = []

        # AI-generation risk
        if is_fake:
            score = max(score, 2)

            if not severe_hits and not conflict_hits:
                reasons.append(
                    f"The image is predicted to be AI-generated with confidence {final_confidence:.2f}. "
                    f"Because no dangerous content is confirmed, this only raises the score to low/moderate risk."
                )
            else:
                reasons.append(
                    f"The image is predicted to be AI-generated with confidence {final_confidence:.2f}."
                )

        # Person / face risk
        if is_fake and contains_person:
            score = max(score, 3)
            reasons.append(
                f"The image appears AI-generated and contains a person/face: "
                f"{self._format_matches(context_hits)}."
            )

        # Conflict / disaster content
        if conflict_hits:
            score = max(score, 4)
            reasons.append(
                f"Conflict or disaster indicators detected: "
                f"{self._format_matches(conflict_hits)}."
            )

        # Severe-risk objects
        if severe_hits:
            score = max(score, 5)
            reasons.append(
                f"Severe-risk objects detected: "
                f"{self._format_matches(severe_hits)}."
            )

        # Combined risk escalation
        if is_fake and conflict_hits:
            score = max(score, 5)
            reasons.append(
                "AI-generation signals combined with conflict/disaster content increase the risk."
            )

        if is_fake and severe_hits:
            score = 5
            reasons.append(
                "AI-generation signals combined with weapons or severe-risk content produce maximum risk."
            )

        if contains_person and severe_hits:
            score = 5
            reasons.append(
                "A person/face appears together with severe-risk objects."
            )

        # Detector disagreement
        detector_gap = abs(general_fake_score - human_fake_score)

        if human_fake_score > 0 and detector_gap >= 0.60:
            score = max(score, 2)
            reasons.append(
                f"The AI detectors disagree "
                f"(general={general_fake_score:.2f}, human={human_fake_score:.2f}), "
                f"so the result should be interpreted carefully."
            )
            
        # Low-risk fallback
        if not reasons:
            reasons.append(
                "No strong AI-generation, person-related, weapon, or conflict indicators were detected."
            )

        explanation = "\n".join(f"- {reason}" for reason in reasons)

        return score, explanation