# Train the anemia model using ONLY the Sickle Cell images
# ----------------------------------------------------------
# This script mirrors `train_with_labels.py` but restricts the dataset
# to the `sickle cell` folder and saves the fine‑tuned weights to a
# separate file (`M_Anemia_finetuned_sickle.h5`) so the generic
# fine‑tuned model is not overwritten.

import os
import random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

# Project imports
from model_service import load_anemia_model, CLASSES, reset_model_cache

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 4
EPOCHS = 30
LR_PHASE1 = 1e-4  # heads only
LR_PHASE2 = 5e-5  # full fine‑tuning
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------------------------------------
# Map folder name to class name (only Sickle Cell here)
# ----------------------------------------------------------
FOLDER_CLASS_MAP = {
    "sickle cell": "Sickle Cell Anemia",
}

def get_class_idx(folder_name: str) -> int:
    """Return the integer index for the disease represented by *folder_name*."""
    disease = FOLDER_CLASS_MAP.get(folder_name)
    if disease is None:
        raise ValueError(f"Unknown training folder: {folder_name}")
    return CLASSES.index(disease)

# ----------------------------------------------------------
# Simple dataset implementation
# ----------------------------------------------------------
class SimpleAnemiaDataset(Dataset):
    """Loads images from a list and returns (tensor, class_idx)."""
    def __init__(self, image_paths, class_idxs, transform=None):
        self.image_paths = image_paths
        self.class_idxs = class_idxs
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.class_idxs[idx]

# ----------------------------------------------------------
# Gather data – ONLY the sickle‑cell folder
# ----------------------------------------------------------
image_paths = []
class_idxs = []

for folder in ["sickle cell"]:
    base = Path(folder)
    cls_idx = get_class_idx(folder)
    for img_file in sorted(base.glob("*")):
        if img_file.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}:
            image_paths.append(str(img_file))
            class_idxs.append(cls_idx)

print("Collected training data:")
print(f"  Total images : {len(image_paths)}")
print(f"  Sickle Cell   : {class_idxs.count(CLASSES.index('Sickle Cell Anemia'))}")

# ----------------------------------------------------------
# Data augmentation (same as the original script)
# ----------------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.3),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.12, 0.12), scale=(0.85, 1.15)),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ----------------------------------------------------------
# Train/validation split (20 % validation)
# ----------------------------------------------------------
full_dataset = SimpleAnemiaDataset(image_paths, class_idxs, transform=train_transform)
val_dataset = SimpleAnemiaDataset(image_paths, class_idxs, transform=val_transform)

val_len = int(0.2 * len(full_dataset))
train_len = len(full_dataset) - val_len
train_set, _ = random_split(full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(SEED))
val_set, _ = random_split(val_dataset, [val_len, train_len], generator=torch.Generator().manual_seed(SEED))

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ----------------------------------------------------------
# Load the base checkpoint
# ----------------------------------------------------------
model = load_anemia_model(h5_path='M_Anemia.h5').to(DEVICE)
print("Loaded M_Anemia.h5 as starting checkpoint")

# ----------------------------------------------------------
# Phase 1 – train only the dense heads (freeze conv blocks)
# ----------------------------------------------------------
for param in model.features.parameters():
    param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_PHASE1)

best_acc = 0.0
print("\n--- Phase 1: Training heads only (epochs 1‑10) ---")
for epoch in range(1, 11):
    model.train()
    epoch_loss = 0.0
    for imgs, labs in train_loader:
        imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
        optimizer.zero_grad()
        outs = model(imgs)
        loss = criterion(outs, labs)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * imgs.size(0)
    epoch_loss /= len(train_loader.dataset)

    # validation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labs in val_loader:
            imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
            outs = model(imgs)
            _, preds = torch.max(outs, 1)
            correct += (preds == labs).sum().item()
            total += labs.size(0)
    acc = correct / total if total else 0.0
    print(f"Epoch {epoch:02d} | loss {epoch_loss:.4f} | val-acc {acc*100:.1f}%")
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "M_Anemia_finetuned_sickle.h5")

# ----------------------------------------------------------
# Phase 2 – unfreeze everything and fine‑tune
# ----------------------------------------------------------
for param in model.features.parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam(model.parameters(), lr=LR_PHASE2)
print("\n--- Phase 2: Full fine‑tuning (epochs 11‑30) ---")
for epoch in range(11, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for imgs, labs in train_loader:
        imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
        optimizer.zero_grad()
        outs = model(imgs)
        loss = criterion(outs, labs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item() * imgs.size(0)
    epoch_loss /= len(train_loader.dataset)

    # validation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labs in val_loader:
            imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
            outs = model(imgs)
            _, preds = torch.max(outs, 1)
            correct += (preds == labs).sum().item()
            total += labs.size(0)
    acc = correct / total if total else 0.0
    print(f"Epoch {epoch:02d} | loss {epoch_loss:.4f} | val-acc {acc*100:.1f}%")
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "M_Anemia_finetuned_sickle.h5")

print("\nTraining complete! Best validation accuracy: {:.1f}%".format(best_acc * 100))
print("Fine‑tuned weights saved to M_Anemia_finetuned_sickle.h5")
reset_model_cache()
print("Model cache cleared.")

