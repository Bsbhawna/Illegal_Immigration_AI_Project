# src/fake_document_detection/signature_verification/train_siamese.py

import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from siamese_model import SiameseNetwork
from signature_dataset import SignatureDataset

# === CONFIG ===
#DATA_PATH = "C:/Users/sharmbha/Downloads/Bhawna_project/Illegal_Immigration_AI_Project/src/fake_document_detection/signature_verification/data"
#CHECKPOINT_PATH = "signature_siamese.pth"
#FINAL_MODEL_PATH = "C:/Users/sharmbha/Downloads/Bhawna_project/Illegal_Immigration_AI_Project/models/siamese_signature_model.pth"
#LOSS_LOG_PATH = "loss_log.txt"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # directory of current script

DATA_PATH = os.path.join(BASE_DIR, "data")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "signature_siamese.pth")
FINAL_MODEL_PATH = os.path.join(BASE_DIR, "../../models/siamese_signature_model.pth")  # adjust relative path as per your structure
LOSS_LOG_PATH = os.path.join(BASE_DIR, "loss_log.txt")

# === Contrastive Loss Definition ===
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        # label = 1 for genuine (similar), 0 for forged (dissimilar)
        loss_similar = label * torch.pow(distance, 2)
        loss_dissimilar = (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        loss = torch.mean(loss_similar + loss_dissimilar)
        return loss

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📟 Using device: {device}")

    dataset = SignatureDataset(DATA_PATH)
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss(margin=1.0)  # Use ContrastiveLoss here
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    start_epoch = 0

    if RESUME and os.path.exists(CHECKPOINT_PATH):
        try:
            print("🔄 Resuming from checkpoint...")
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            print(f"✅ Resumed from epoch {start_epoch}")
        except Exception as e:
            print(f"⚠️ Error loading checkpoint: {e}. Starting fresh.")
    else:
        print("🚀 Starting fresh training...")

    if start_epoch >= EPOCHS:
        print("✅ Training already completed previously. Nothing to resume.")
        if not os.path.exists(FINAL_MODEL_PATH):
            os.makedirs(os.path.dirname(FINAL_MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), FINAL_MODEL_PATH)
            print("💾 Final model was not saved before — saving now.")
        else:
            print("📁 Final model already exists.")
        return

    try:
        from tqdm import tqdm
        for epoch in range(start_epoch, EPOCHS):
            print(f"\n🔁 Training epoch {epoch+1}/{EPOCHS}...")
            model.train()
            total_loss = 0.0

            for batch_idx, (img1, img2, label) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"🌀 Epoch {epoch+1}"):
                img1, img2 = img1.to(device), img2.to(device)
                label = label.float().to(device)  # label shape: [batch_size]

                output1, output2 = model(img1, img2)
                distance = torch.nn.functional.pairwise_distance(output1, output2)

                loss = criterion(distance, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"✅ Epoch {epoch+1}/{EPOCHS} completed. Loss: {avg_loss:.4f}")

            with open(LOSS_LOG_PATH, "a") as f:
                f.write(f"{epoch+1},{avg_loss:.4f}\n")

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, CHECKPOINT_PATH)

    except KeyboardInterrupt:
        print("\n⛔ Training interrupted by user. Saving checkpoint...")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, CHECKPOINT_PATH)
        print(f"✅ Checkpoint saved at epoch {epoch}. Resume later safely.")

    if epoch + 1 == EPOCHS:
        torch.save(model.state_dict(), FINAL_MODEL_PATH)
        print("🎉 Training complete! Final model saved.")

if __name__ == "__main__":
    train()
