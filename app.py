import gradio as gr
from PIL import Image, ImageDraw
import numpy as np

from ai_detector import AIDetector
from human_detector import HumanDetector
from object_detector import ObjectDetector
from risk_engine import RiskEngine
from utils import ensure_pil_image


GENERAL_MODEL_PATH = "models/improved_AI_Generated.pt"
HUMAN_MODEL_PATH = "models/ai_image_detector_model_improved_convnext.pth"


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


def draw_placeholder(text):
    img = Image.new("RGB", (500, 320), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.text((30, 135), text, fill="black")
    return img


def draw_boxes(image, detected_objects):
    image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)

    for obj in detected_objects:
        box = obj.get("box")
        if not box:
            continue

        label = str(obj.get("label", "unknown"))
        confidence = float(obj.get("confidence", 0.0))
        x1, y1, x2, y2 = [int(v) for v in box]

        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)

        text = f"{label} ({confidence:.2f})"
        bbox = draw.textbbox((x1, y1), text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        draw.rectangle(
            [x1, max(0, y1 - text_h - 8), x1 + text_w + 8, y1],
            fill="red"
        )
        draw.text((x1 + 4, max(0, y1 - text_h - 6)), text, fill="white")

    return image


def get_risk_label(score):
    if score == 1:
        return "🟢 1/5 - Low Risk"
    if score == 2:
        return "🔵 2/5 - Mild Concern"
    if score == 3:
        return "🟠 3/5 - Moderate Risk"
    if score == 4:
        return "🔴 4/5 - High Risk"
    return "🛑 5/5 - Severe Risk"


def split_objects(detected_objects):
    risk_terms = {
        "gun", "pistol", "handgun", "rifle",
        "knife", "fire", "flames", "smoke",
        "explosion", "tank", "military vehicle", "helicopter"
    }

    risk = []
    context = []

    for obj in detected_objects:
        label = str(obj.get("label", "")).lower()
        conf = float(obj.get("confidence", 0.0))
        line = f"- {label} ({conf:.2f})"

        if any(term in label for term in risk_terms):
            risk.append(line)
        else:
            context.append(line)

    risk_text = "\n".join(risk) if risk else "No high-risk objects detected."
    context_text = "\n".join(context) if context else "No context objects detected."

    return risk_text, context_text


def make_ai_heatmap(detector, image, target_label, grid_size=6):
    image = ensure_pil_image(image).convert("RGB")
    small = image.resize((224, 224))

    _, _, base_probs = detector.predict(small)

    if target_label not in base_probs:
        return draw_placeholder(f"Heatmap unavailable:\n'{target_label}' not found in model outputs.")

    base_score = float(base_probs[target_label])
    heat = np.zeros((grid_size, grid_size), dtype=np.float32)

    cell_w = 224 // grid_size
    cell_h = 224 // grid_size

    for row in range(grid_size):
        for col in range(grid_size):
            masked = small.copy()
            draw = ImageDraw.Draw(masked)

            x1 = col * cell_w
            y1 = row * cell_h
            x2 = 224 if col == grid_size - 1 else (col + 1) * cell_w
            y2 = 224 if row == grid_size - 1 else (row + 1) * cell_h

            draw.rectangle([x1, y1, x2, y2], fill=(128, 128, 128))

            _, _, new_probs = detector.predict(masked)
            new_score = float(new_probs.get(target_label, 0.0))

            drop = max(0.0, base_score - new_score)
            heat[row, col] = drop

    if heat.max() > 0:
        heat = heat / heat.max()
    else:
        heat = np.ones((grid_size, grid_size), dtype=np.float32) * 0.25

    heat_img = Image.fromarray(np.uint8(heat * 255), mode="L")
    heat_img = heat_img.resize(image.size, Image.Resampling.BILINEAR)

    red_overlay = Image.new("RGBA", image.size, (255, 0, 0, 0))
    alpha = heat_img.point(lambda p: int(min(255, p * 1.5)))
    red_overlay.putalpha(alpha)

    base = image.convert("RGBA")
    combined = Image.alpha_composite(base, red_overlay)

    return combined.convert("RGB")


def magnify_click(image, evt: gr.SelectData):
    if image is None:
        return None

    image = ensure_pil_image(image).convert("RGB")
    x, y = evt.index

    crop_size = 80
    zoom_size = 450

    left = max(0, x - crop_size // 2)
    upper = max(0, y - crop_size // 2)
    right = min(image.width, x + crop_size // 2)
    lower = min(image.height, y + crop_size // 2)

    crop = image.crop((left, upper, right, lower))
    crop = crop.resize((zoom_size, zoom_size), Image.Resampling.NEAREST)

    draw = ImageDraw.Draw(crop)
    mid = zoom_size // 2

    draw.line([(mid, 0), (mid, zoom_size)], fill="red", width=2)
    draw.line([(0, mid), (zoom_size, mid)], fill="red", width=2)

    return crop


def show_loading():
    return (
        """
## ⏳ Analysing image...

Please wait while the system runs:

- General AI detector  
- Human-focused detector if a person or face is detected  
- Grounding DINO object detector  
- Risk scoring engine  
- Optional AI heatmap generation  

### 🔍 🤖 ⚙️
""",
        draw_placeholder("Running object detection..."),
        "Analysis running...",
        draw_placeholder("Generating general AI heatmap..."),
        draw_placeholder("Generating human detector heatmap..."),
        "Analysis running..."
    )


def analyse_image(image, generate_heatmaps):
    if image is None:
        return (
            "## No image uploaded\nPlease upload an image first.",
            draw_placeholder("No image uploaded."),
            "No objects detected.",
            draw_placeholder("No image uploaded."),
            draw_placeholder("No image uploaded."),
            "No technical output."
        )

    image = ensure_pil_image(image).convert("RGB")

    detected_objects = object_detector.detect_objects(image)
    labels = [obj["label"].lower() for obj in detected_objects]

    contains_person = any(
        "person" in label or "face" in label
        for label in labels
    )

    general_pred, general_conf, general_probs = general_detector.predict(image)

    general_fake_score = (
        general_probs.get("AI_GENERATED")
        or general_probs.get("fake")
        or general_probs.get("FAKE")
        or 0.0
    )

    if contains_person:
        human_pred, human_conf, human_probs = human_detector.predict(image)
        human_fake_score = (
            human_probs.get("fake")
            or human_probs.get("FAKE")
            or human_probs.get("AI_GENERATED")
            or 0.0
        )
    else:
        human_pred = "NOT USED"
        human_conf = 0.0
        human_fake_score = 0.0

    if contains_person:
        final_fake_score = (general_fake_score * 0.75) + (human_fake_score * 0.25)
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

    risk_objects, context_objects = split_objects(detected_objects)
    annotated_image = draw_boxes(image, detected_objects)

    summary = f"""
## Result Summary

**Final Prediction:** {final_prediction.upper()}  
**Confidence:** {final_confidence:.4f}  
**Risk Score:** {get_risk_label(risk_score)}

### Explanation
{explanation}
"""

    object_summary = f"""
## Risk Objects
{risk_objects}

## Context Objects
{context_objects}
"""

    technical = (
        f"General Model Prediction: {general_pred.upper()} ({general_conf:.4f})\n"
        f"Human Model Prediction: {str(human_pred).upper()} ({human_conf:.4f})\n"
        f"General Fake Score: {general_fake_score:.4f}\n"
        f"Human Fake Score: {human_fake_score:.4f}\n"
        f"Final Combined Prediction: {final_prediction.upper()}\n"
        f"Final Combined Confidence: {final_confidence:.4f}\n"
        f"Risk Score: {risk_score}/5"
    )

    if generate_heatmaps:
        general_heatmap = make_ai_heatmap(
            detector=general_detector,
            image=image,
            target_label="AI_GENERATED",
            grid_size=6
        )

        if contains_person:
            human_heatmap = make_ai_heatmap(
                detector=human_detector,
                image=image,
                target_label="fake",
                grid_size=6
            )
        else:
            human_heatmap = draw_placeholder(
                "Human detector heatmap not used\nbecause no person/face was detected."
            )
    else:
        general_heatmap = draw_placeholder(
            "Enable 'Generate AI heatmaps'\nto see model reasoning."
        )
        human_heatmap = draw_placeholder(
            "Enable 'Generate AI heatmaps'\nto see model reasoning."
        )

    return (
        summary,
        annotated_image,
        object_summary,
        general_heatmap,
        human_heatmap,
        technical
    )


with gr.Blocks(title="Layered AI Image Detection and Risk Scoring System") as demo:
    gr.Markdown(
        "<h1 style='text-align: center;'>Layered AI Image Detection and Risk Scoring System</h1>"
    )
    gr.Markdown(
        "<p style='text-align: center;'>Upload an image to analyse whether it is AI-generated and assess its moderation risk.</p>"
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="pil",
                label="Upload Image",
                height=520,
                width=520,
                image_mode="RGB"
            )

            analyse_button = gr.Button("Analyse Image", variant="primary")

            generate_heatmaps = gr.Checkbox(
                label="Generate AI heatmaps (slower)",
                value=True
            )

            gr.Markdown(
                "**Prototype note:** This system supports moderation decisions but should not replace human review."
            )

        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.Tab("Summary"):
                    summary_output = gr.Markdown()

                with gr.Tab("Detected Objects"):
                    annotated_output = gr.Image(
                        label="Detected Object Regions",
                        height=450
                    )
                    object_summary_output = gr.Markdown()

                with gr.Tab("AI Heatmaps"):
                    gr.Markdown(
                        """
Heatmaps show which image regions influenced the AI detector.  
Redder areas had more influence on the model output.  
This does **not** prove those regions are fake.
"""
                    )

                    general_heatmap_output = gr.Image(
                        label="General AI Detector Heatmap",
                        height=350
                    )

                    human_heatmap_output = gr.Image(
                        label="Human Detector Heatmap",
                        height=350
                    )

                with gr.Tab("Pixel Magnifier"):
                    gr.Markdown(
                        "Click a point on the uploaded image to inspect it at pixel level."
                    )
                    magnifier_output = gr.Image(
                        label="Magnified Pixel View",
                        height=450
                    )

                with gr.Tab("Technical Details"):
                    technical_output = gr.Textbox(
                        label="Technical Output",
                        lines=15,
                        max_lines=15,
                        autoscroll=False
                    )

                with gr.Tab("About"):
                    gr.Markdown(
                        """
## About this prototype

This prototype uses:

1. **General AI detector** - predicts whether the whole image is real or AI-generated.  
2. **Human-focused detector** - used only when a person or face is detected.  
3. **Grounding DINO** - identifies objects and risk indicators.  
4. **Risk engine** - combines prediction confidence, object detection and context.  
5. **Heatmaps** - show which regions influenced AI classification decisions.

### Risk score guide

- **1/5:** Low risk  
- **2/5:** AI-generated but low-risk  
- **3/5:** Moderate risk  
- **4/5:** High risk  
- **5/5:** Severe risk  

This is a research prototype and can still make mistakes.
"""
                    )

    analyse_button.click(
        fn=show_loading,
        inputs=None,
        outputs=[
            summary_output,
            annotated_output,
            object_summary_output,
            general_heatmap_output,
            human_heatmap_output,
            technical_output
        ]
    ).then(
        fn=analyse_image,
        inputs=[input_image, generate_heatmaps],
        outputs=[
            summary_output,
            annotated_output,
            object_summary_output,
            general_heatmap_output,
            human_heatmap_output,
            technical_output
        ]
    )

    input_image.select(
        fn=magnify_click,
        inputs=input_image,
        outputs=magnifier_output
    )


if __name__ == "__main__":
    demo.launch()