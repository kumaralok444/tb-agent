import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import sys

from model import build_model, get_device
from dataset import EVAL_TRANSFORMS

CHECKPOINT = Path("checkpoints/best_model.pth")

# TB probability thresholds
GREEN_MAX = 0.30   # < 30%  -> TB Negative
YELLOW_MAX = 0.70  # 30-70% -> Uncertain
                   # > 70%  -> TB Positive


def load_model(device):
    model = build_model(num_classes=2)
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"No checkpoint found at {CHECKPOINT}. Train the model first.")
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.to(device)
    model.eval()
    return model


def classify(image_path: str, model, device) -> dict:
    image = Image.open(image_path).convert("RGB")
    tensor = EVAL_TRANSFORMS(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    tb_prob = probs[1].item()
    normal_prob = probs[0].item()

    if tb_prob < GREEN_MAX:
        status = "GREEN"
        label = "TB Negative"
        message = "No signs of TB detected."
    elif tb_prob < YELLOW_MAX:
        status = "YELLOW"
        label = "Uncertain"
        message = "Inconclusive result. Clinical review recommended."
    else:
        status = "RED"
        label = "TB Positive"
        message = "High likelihood of TB. Immediate clinical attention advised."

    return {
        "status": status,
        "label": label,
        "tb_probability": round(tb_prob * 100, 2),
        "normal_probability": round(normal_prob * 100, 2),
        "message": message,
    }


def print_result(result: dict):
    colors = {"RED": "\033[91m", "YELLOW": "\033[93m", "GREEN": "\033[92m"}
    reset = "\033[0m"
    color = colors[result["status"]]

    print("\n" + "=" * 50)
    print(f"{color}  STATUS : {result['status']} — {result['label']}{reset}")
    print(f"  TB Prob: {result['tb_probability']}%")
    print(f"  Normal : {result['normal_probability']}%")
    print(f"  Note   : {result['message']}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    device = get_device()
    model = load_model(device)
    result = classify(image_path, model, device)
    print_result(result)
