import os
import gradio as gr

from ai_detector import AIDetector
from human_detector import HumanDetector
from object_detector import ObjectDetector
from risk_engine import RiskEngine
from utils import ensure_pil_image


# PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

GENERAL_MODEL_PATH = os.path.join(MODELS_DIR, "ai_image_detector_model.pth")
HUMAN_MODEL_PATH = os.path.join(MODELS_DIR, "ai_image_detector_model_improved_convnext.pth")
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n.pt")


# LOAD MODELS
general_detector = AIDetector(
    model_path=GENERAL_MODEL_PATH,
    class_names=["fake", "real"]
)

human_detector = HumanDetector(
    model_path=HUMAN_MODEL_PATH,
    class_names=["fake", "real"]
)

object_detector = ObjectDetector(model_path=YOLO_MODEL_PATH)
risk_engine = RiskEngine()


# MAIN ANALYSIS FUNCTION
def analyse_image(image):
    image = ensure_pil_image(image)

    # Run object detector
    detected_objects = object_detector.detect_objects(image)
    labels = [obj["label"] for obj in detected_objects]

    # Run general model
    general_pred, general_conf, general_probs = general_detector.predict(image)
    general_fake_score = general_probs["fake"]

    # Run human-focused model on all images as second layer
    human_pred, human_conf, human_probs = human_detector.predict(image)
    human_fake_score = human_probs["fake"]

    # Combine both AI detector scores
    # Human model has extra weight because the improved dataset is harder and person-focused
    if "person" in labels:
        final_fake_score = (general_fake_score * 0.4) + (human_fake_score * 0.6)
    else:
        final_fake_score = (general_fake_score * 0.6) + (human_fake_score * 0.4)

    # Convert combined fake score to final prediction
    if final_fake_score >= 0.5:
        final_prediction = "fake"
        final_confidence = final_fake_score
    else:
        final_prediction = "real"
        final_confidence = 1 - final_fake_score

    # Risk scoring
    risk_score, explanation = risk_engine.calculate_risk(
    image=image,
    final_prediction=final_prediction,
    final_confidence=final_confidence,
    detected_objects=detected_objects,
    general_fake_score=general_fake_score,
    human_fake_score=human_fake_score
    )
    

    # Format object output
    if detected_objects:
        object_text = "\n".join(
            [f"- {obj['label']} ({obj['confidence']:.2f})" for obj in detected_objects]
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


# INTERFACE
demo = gr.Interface(
    fn=analyse_image,
    inputs=gr.Image(type="pil", label="Upload or drag-and-drop an image"),
    outputs=gr.Textbox(label="Analysis Result"),
    title="Layered AI Image Detection and Risk Scoring System",
    description="This prototype uses a general AI detector, a human-focused AI detector, and an object detector before assigning a final moderation risk score."
)


# RUN APP
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
