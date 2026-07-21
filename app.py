import gradio as gr
from pathlib import Path
from predict import load_model, classify
from model import get_device

device = get_device()
model = load_model(device)


def analyze_image(image):
    import tempfile, os
    from PIL import Image as PILImage

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        PILImage.fromarray(image).save(tmp.name)
        result = classify(tmp.name, model, device)
        os.unlink(tmp.name)

    color_map = {"RED": "#ff4444", "YELLOW": "#ffaa00", "GREEN": "#44bb44"}
    color = color_map[result["status"]]

    html = f"""
    <div style="border-radius:12px; padding:20px; background:{color}22; border: 3px solid {color}; font-family:sans-serif;">
        <h2 style="color:{color}; margin:0 0 12px 0;">⬤ {result['status']} — {result['label']}</h2>
        <p style="font-size:16px; margin:4px 0;"><b>TB Probability:</b> {result['tb_probability']}%</p>
        <p style="font-size:16px; margin:4px 0;"><b>Normal Probability:</b> {result['normal_probability']}%</p>
        <hr style="border-color:{color}44; margin:12px 0;">
        <p style="font-size:15px; color:#333; margin:0;">📋 {result['message']}</p>
    </div>
    """
    return html


with gr.Blocks(title="TB Detection Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🫁 TB Detection Agent\nUpload a chest X-ray image to analyze for Tuberculosis.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Chest X-Ray", type="numpy")
            analyze_btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            result_output = gr.HTML(label="Result")

    analyze_btn.click(fn=analyze_image, inputs=image_input, outputs=result_output)

    gr.Markdown("""
    ---
    **Color Guide:**
    - 🟢 **GREEN** — TB Negative (< 30% probability)
    - 🟡 **YELLOW** — Uncertain (30–70% probability) — clinical review advised
    - 🔴 **RED** — TB Positive (> 70% probability) — immediate attention needed

    > ⚠️ This tool is for research/screening purposes only. Always consult a qualified medical professional.
    """)

if __name__ == "__main__":
    demo.launch(share=False)
