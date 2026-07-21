# TB Detection Agent

Offline TB detection from chest X-ray images using EfficientNet-B3 + PyTorch.

## Setup

```bash
cd tb-agent
pip install -r requirements.txt
```

## 1. Prepare Your Data

Put images into these folders:
```
data/
├── tb/        ← chest X-rays WITH tuberculosis
└── normal/    ← chest X-rays WITHOUT tuberculosis
```
Supports: `.jpg`, `.jpeg`, `.png`, `.bmp`

## 2. Train the Model

```bash
python train.py
```

- Trains for 30 epochs on Apple Silicon (MPS)
- Best model saved to `checkpoints/best_model.pth`
- Takes ~1–2 hours for 5000+ images

## 3. Run Predictions

**Terminal (single image):**
```bash
python predict.py path/to/xray.jpg
```

**Browser UI (Gradio):**
```bash
python app.py
```
Then open `http://localhost:7860`

## Color Guide

| Color  | Meaning       | TB Probability |
|--------|---------------|----------------|
| 🟢 GREEN  | TB Negative   | < 30%         |
| 🟡 YELLOW | Uncertain     | 30–70%        |
| 🔴 RED    | TB Positive   | > 70%         |

## Public Datasets (if you need training data)

- [NIH Chest X-ray Dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data)
- [Montgomery & Shenzhen TB Dataset](https://www.kaggle.com/datasets/kmader/pulmonary-chest-xray-abnormalities)

> ⚠️ For research/screening only. Not a substitute for clinical diagnosis.
