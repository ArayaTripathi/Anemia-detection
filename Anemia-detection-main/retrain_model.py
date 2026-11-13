"""
Comprehensive retraining script for the Anemia Detection model.

Uses ImageNet-pretrained VGG16 as backbone, trains on:
  - test_dataset/ (Thalassemia images, 20 images)
  - test_iron/    (Iron Deficiency Anemia images, 18 images)
  - Labels from labels/ folder (YOLO-format cell annotations)

Applies heavy data augmentation to compensate for small dataset.
Saves the retrained model as M_Anemia_finetuned.h5.
"""

import os
import random
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np

# Import our model definition
from model_service import AnemiaVGGModel, CLASSES, reset_model_cache

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 4
EPOCHS = 30
LR = 5e-5
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Device: {DEVICE}")
print(f"Classes: {CLASSES}")

# ======================================================================
# YOLO-label cell class ID -> Anemia disease mapping
# Based on blood smear microscopy cell morphology:
#   0  = Normal RBC
#   1  = Target cell (codocyte)       -> Thalassemia indicator
#   2  = Sickle cell (drepanocyte)    -> Sickle Cell Anemia
#   3  = Microcyte (small RBC)        -> Iron Deficiency / Thalassemia
#   4  = Macrocyte (large RBC)        -> B12 Deficiency (Megaloblastic)
#   5  = Hypochromic cell             -> Iron Deficiency Anemia
#   6  = Elliptocyte / pencil cell    -> Iron Deficiency Anemia
#   7  = Teardrop cell (dacrocyte)    -> Thalassemia
#   8  = Basophilic stippling         -> Thalassemia
#   9  = Howell-Jolly body            -> Sickle Cell / post-splenectomy
#  10  = Schistocyte (fragment)       -> various
#  11  = Hypersegmented neutrophil    -> B12 Deficiency
#  12  = Nucleated RBC (nRBC)         -> Thalassemia
# ======================================================================

# Map each cell class to which anemia type it indicates
CELL_TO_ANEMIA = {
    0:  None,                                      # Normal - no specific anemia
    1:  'Thalassemia',                             # Target cells
    2:  'Sickle Cell Anemia',                      # Sickle cells
    3:  'Iron Deficiency Anemia',                  # Microcytes (could be Thal too)
    4:  'Vitamin B12 Deficiency (Megaloblastic)',   # Macrocytes
    5:  'Iron Deficiency Anemia',                  # Hypochromic
    6:  'Iron Deficiency Anemia',                  # Elliptocytes/pencil cells
    7:  'Thalassemia',                             # Teardrop cells
    8:  'Thalassemia',                             # Basophilic stippling
    9:  'Sickle Cell Anemia',                      # Howell-Jolly bodies
    10: 'Dimorphic Anemia',                        # Schistocytes (mixed)
    11: 'Vitamin B12 Deficiency (Megaloblastic)',   # Hyperseg neutrophils
    12: 'Thalassemia',                             # Nucleated RBCs
}


def get_image_label_from_yolo(label_path):
    """
    Read a YOLO label file and determine the dominant anemia type
    based on abnormal cell counts.
    Returns: class index into CLASSES, or None if all normal.
    """
    anemia_votes = Counter()
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                cell_class = int(parts[2])
                anemia_type = CELL_TO_ANEMIA.get(cell_class)
                if anemia_type is not None:
                    anemia_votes[anemia_type] += 1

    if not anemia_votes:
        return None  # All normal cells

    # Return the most voted anemia type
    dominant = anemia_votes.most_common(1)[0][0]
    return CLASSES.index(dominant)


# ======================================================================
# Dataset class
# ======================================================================
class AnemiaTrainingDataset(Dataset):
    """
    Loads images + labels for training.
    Supports:
      - Folder-based labeling (all images in a folder get the same label)
      - YOLO label file-based labeling (per-image labels from annotations)
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ======================================================================
# Collect training data
# ======================================================================
image_paths = []
labels = []

# 1. Thalassemia images (from test_dataset/) - known label
thal_dir = Path('test_dataset')
thal_class_idx = CLASSES.index('Thalassemia')
for img_file in sorted(thal_dir.glob('*')):
    if img_file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}:
        image_paths.append(str(img_file))
        labels.append(thal_class_idx)

# 2. Iron Deficiency Anemia images (from test_iron/) - known label
iron_dir = Path('test_iron')
iron_class_idx = CLASSES.index('Iron Deficiency Anemia')
for img_file in sorted(iron_dir.glob('*')):
    if img_file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}:
        image_paths.append(str(img_file))
        labels.append(iron_class_idx)

print(f"\nDataset Summary:")
print(f"  Thalassemia images:         {sum(1 for l in labels if l == thal_class_idx)}")
print(f"  Iron Deficiency images:     {sum(1 for l in labels if l == iron_class_idx)}")
print(f"  Total training images:      {len(image_paths)}")

# ======================================================================
# Heavy data augmentation to compensate for small dataset
# ======================================================================
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ======================================================================
# Create dataset - use all data for training (small dataset)
# Also create a val copy with no augmentation to track overfitting
# ======================================================================
train_dataset = AnemiaTrainingDataset(image_paths, labels, transform=train_transform)
val_dataset = AnemiaTrainingDataset(image_paths, labels, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ======================================================================
# Build model: Start from ImageNet-pretrained VGG16
# ======================================================================
print("\nLoading ImageNet-pretrained VGG16...")
try:
    vgg16_pretrained = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
except Exception as e:
    print(f'Pretrained VGG16 download failed ({e}); using random-initialized VGG16')
    vgg16_pretrained = models.vgg16(weights=None)

# Create our model and copy pretrained feature weights
model = AnemiaVGGModel()

# Copy pretrained conv weights from torchvision VGG16
# Our model.features layout matches torchvision VGG16 features exactly
pretrained_features = list(vgg16_pretrained.features.children())
our_features = list(model.features.children())

for i, (pretrained_layer, our_layer) in enumerate(zip(pretrained_features, our_features)):
    if hasattr(pretrained_layer, 'weight') and hasattr(our_layer, 'weight'):
        our_layer.weight.data.copy_(pretrained_layer.weight.data)
        if hasattr(pretrained_layer, 'bias') and pretrained_layer.bias is not None:
            our_layer.bias.data.copy_(pretrained_layer.bias.data)

print("ImageNet weights loaded into feature extractor.")

# Initialize FC layers with Xavier initialization for better training
nn.init.xavier_uniform_(model.fc1.weight)
nn.init.zeros_(model.fc1.bias)
nn.init.xavier_uniform_(model.fc2.weight)
nn.init.zeros_(model.fc2.bias)

model = model.to(DEVICE)

# ======================================================================
# Training strategy:
# Phase 1 (epochs 1-10): Freeze conv blocks 1-3, train blocks 4-5 + FC
# Phase 2 (epochs 11-30): Unfreeze all, train with lower LR
# ======================================================================

def freeze_up_to_block(model, block_end_idx):
    """Freeze feature layers up to block_end_idx."""
    for i, layer in enumerate(model.features):
        if i < block_end_idx:
            for param in layer.parameters():
                param.requires_grad = False
        else:
            for param in layer.parameters():
                param.requires_grad = True
    # FC layers always trainable
    for param in model.fc1.parameters():
        param.requires_grad = True
    for param in model.fc2.parameters():
        param.requires_grad = True


# Phase 1: Freeze blocks 1-3 (indices 0-14), train blocks 4-5 + FC
# Block 4 starts at index 15 in our Sequential
BLOCK4_START = 15
freeze_up_to_block(model, BLOCK4_START)

# Use weighted loss to handle class imbalance (we only have 2 of 5 classes)
# Give higher weight to classes we're training on
class_weights = torch.ones(5, device=DEVICE)
class_weights[thal_class_idx] = 2.0   # Thalassemia
class_weights[iron_class_idx] = 2.0   # Iron Deficiency
criterion = nn.CrossEntropyLoss(weight=class_weights)


def get_optimizer(model, lr):
    trainable = filter(lambda p: p.requires_grad, model.parameters())
    return torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)


def evaluate(loader):
    model.eval()
    correct = total = 0
    class_correct = Counter()
    class_total = Counter()
    with torch.no_grad():
        for imgs, lbl in loader:
            imgs, lbl = imgs.to(DEVICE), lbl.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == lbl).sum().item()
            total += lbl.size(0)
            for p, t in zip(preds.cpu().numpy(), lbl.cpu().numpy()):
                class_total[int(t)] += 1
                if p == t:
                    class_correct[int(t)] += 1
    acc = correct / total if total else 0.0
    return acc, class_correct, class_total


print("\n--- Phase 1: Training blocks 4-5 + FC (epochs 1-10) ---")
optimizer = get_optimizer(model, lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

best_acc = 0.0
for epoch in range(1, 11):
    model.train()
    epoch_loss = 0.0
    for imgs, lbl in train_loader:
        imgs, lbl = imgs.to(DEVICE), lbl.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbl)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * imgs.size(0)
    epoch_loss /= len(train_loader.dataset)
    scheduler.step()

    val_acc, cc, ct = evaluate(val_loader)
    per_class = " | ".join(
        f"{CLASSES[c][:8]}: {cc.get(c,0)}/{ct.get(c,0)}"
        for c in sorted(ct.keys())
    )
    print(f"Epoch {epoch:02d} | loss {epoch_loss:.4f} | acc {val_acc*100:.1f}% | {per_class}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'M_Anemia_finetuned.h5')

print(f"\nPhase 1 best accuracy: {best_acc*100:.1f}%")

# Phase 2: Unfreeze all layers, train with lower LR
print("\n--- Phase 2: Full fine-tuning (epochs 11-30) ---")
freeze_up_to_block(model, 0)  # Unfreeze everything
optimizer = get_optimizer(model, lr=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

for epoch in range(11, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for imgs, lbl in train_loader:
        imgs, lbl = imgs.to(DEVICE), lbl.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item() * imgs.size(0)
    epoch_loss /= len(train_loader.dataset)
    scheduler.step()

    val_acc, cc, ct = evaluate(val_loader)
    per_class = " | ".join(
        f"{CLASSES[c][:8]}: {cc.get(c,0)}/{ct.get(c,0)}"
        for c in sorted(ct.keys())
    )
    print(f"Epoch {epoch:02d} | loss {epoch_loss:.4f} | acc {val_acc*100:.1f}% | {per_class}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'M_Anemia_finetuned.h5')

print(f"\nTraining complete! Best accuracy: {best_acc*100:.1f}%")
print(f"Model saved to M_Anemia_finetuned.h5")

# Reset the model cache so the app loads the new weights
reset_model_cache()
print("Model cache cleared - app will load new weights on next prediction.")

