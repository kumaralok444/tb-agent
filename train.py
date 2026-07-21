import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

from model import build_model, get_device
from dataset import get_dataloaders

DATA_DIR = "data"
CHECKPOINT_DIR = Path("checkpoints")
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-4
EARLY_STOP_PATIENCE = 7  # stop if val_loss doesn't improve for this many epochs


def compute_metrics(all_preds, all_labels):
    """Returns accuracy, sensitivity (TB recall), specificity (normal recall)."""
    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
    tn = ((all_preds == 0) & (all_labels == 0)).sum().item()
    fp = ((all_preds == 1) & (all_labels == 0)).sum().item()
    fn = ((all_preds == 0) & (all_labels == 1)).sum().item()

    accuracy    = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)  # true TB recall — most important
    specificity = tn / (tn + fp + 1e-8)  # true normal recall
    return accuracy, sensitivity, specificity


def train():
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)

    model = build_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        # --- training ---
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss /= total

        # --- validation ---
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                all_preds.append(outputs.argmax(dim=1).cpu())
                all_labels.append(labels.cpu())

        val_loss /= sum(len(l) for l in all_labels)
        val_acc, sensitivity, specificity = compute_metrics(all_preds, all_labels)
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"Sensitivity: {sensitivity:.4f} Specificity: {specificity:.4f}"
        )

        # save best model (tracked by val loss, not acc)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            CHECKPOINT_DIR.mkdir(exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pth")
            print(f"  -> Saved best model (val_loss={val_loss:.4f}, sensitivity={sensitivity:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping: no improvement for {EARLY_STOP_PATIENCE} epochs.")
                break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {CHECKPOINT_DIR / 'best_model.pth'}")


if __name__ == "__main__":
    train()
