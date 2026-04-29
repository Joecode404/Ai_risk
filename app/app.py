import os
import gradio as gr

from ai_detector import AIDetector
from human_detector import HumanDetector
from object_detector import ObjectDetector
from risk_engine import RiskEngine
from utils import ensure_pil_image


# -----------------------------
# PATHS
# -----------------------------
GENERAL_MODEL_PATH = "models/improved_AI_Generated.pt"
HUMAN_MODEL_PATH = "models/ai_image_detector_model_improved_convnext.pth"


# -----------------------------
# LOAD MODELS
# -----------------------------
general_detector = AIDetector(
    model_path=GENERAL_MODEL_PATH,
    class_names=["REAL", "AI_GENERATED"]
)

human_detector = HumanDetector(
    model_path=HUMAN_MODEL_PATH,
    class_names=["fake", "real"]
)

object_detector = ObjectDetector()
risk_engine = RiskEngine()


# -----------------------------
# MAIN ANALYSIS FUNCTION
# -----------------------------
def analyse_image(image):
    image = ensure_pil_image(image)

    detected_objects = object_detector.detect_objects(image)
    labels = [obj["label"].lower() for obj in detected_objects]

    general_pred, general_conf, general_probs = general_detector.predict(image)

    general_fake_score = (
        general_probs.get("AI_GENERATED")
        or general_probs.get("fake")
        or general_probs.get("FAKE")
        or 0.0
    )

    human_pred, human_conf, human_probs = human_detector.predict(image)

    human_fake_score = (
        human_probs.get("fake")
        or human_probs.get("FAKE")
        or human_probs.get("AI_GENERATED")
        or 0.0
    )

    if "person" in labels:
        final_fake_score = (general_fake_score * 0.45) + (human_fake_score * 0.55)
    else:
        final_fake_score = general_fake_score

    if final_fake_score >= 0.5:
        final_prediction = "fake"
        final_confidence = final_fake_score
    else:
        final_prediction = "real"
        final_confidence = 1 - final_fake_score

    risk_score, explanation = risk_engine.calculate_risk(
        image=image,
        final_prediction=final_prediction,
        final_confidence=final_confidence,
        detected_objects=detected_objects,
        general_fake_score=general_fake_score,
        human_fake_score=human_fake_score
    )

    if detected_objects:
        object_text = "\n".join(
            f"- {obj['label']} ({obj['confidence']:.2f})"
            for obj in detected_objects
        )
    else:
        object_text = "No objects detected."

    result_text = (
        f"General Model Prediction: {general_pred.upper()} ({general_conf:.4f})\n"
        f"Human Model Prediction: {human_pred.upper()} ({human_conf:.4f})\n"
        f"General Fake Score: {general_fake_score:.4f}\n"
        f"Human Fake Score: {human_fake_score:.4f}\n"
        f"Final Combined Prediction: {final_prediction.upper()}\n"
        f"Final Combined Confidence: {final_confidence:.4f}\n"
        f"Risk Score: {risk_score}/5\n\n"
        f"Detected Objects:\n{object_text}\n\n"
        f"Explanation:\n{explanation}"
    )

    return result_text


# -----------------------------
# INTERFACE
# -----------------------------
demo = gr.Interface(
    fn=analyse_image,
    inputs=gr.Image(type="pil", label="Upload or drag-and-drop an image"),
    outputs=gr.Textbox(label="Analysis Result"),
    title="Layered AI Image Detection and Risk Scoring System",
    description=(
        "This prototype uses a general AI detector, a human-focused AI detector, "
        "Grounding DINO object detection, and a risk engine before assigning a "
        "final moderation risk score."
    )
)


# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    demo.launch()
