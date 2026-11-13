import os
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

# Import model definition and loader from the existing codebase
from model_service import AnemiaVGGModel, load_anemia_model, CLASSES

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------------------------------------------------
# Simple dataset that loads every image in a folder and gives it the same label
# ----------------------------------------------------------------------
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir: str, target_class: str, transform=None):
        self.root = Path(root_dir)
        self.files = sorted([p for p in self.root.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
        self.transform = transform
        self.target = CLASSES.index(target_class)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.target, img_path.name

# ----------------------------------------------------------------------
# Transform matches the preprocessing used in model_service.predict_anemia_sample
# ----------------------------------------------------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),  # values in [0, 1]
])

# ----------------------------------------------------------------------
# Load dataset – all images belong to the "Thalassemia" class
# ----------------------------------------------------------------------
dataset = ImageFolderDataset(root_dir='test_dataset', target_class='Thalassemia', transform=preprocess)

# 20 % validation split
val_len = int(0.2 * len(dataset))
train_len = len(dataset) - val_len
train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(SEED))

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ----------------------------------------------------------------------
# Load the original weights from M_Anemia.h5
# ----------------------------------------------------------------------
model = load_anemia_model(h5_path='M_Anemia.h5').to(DEVICE)

# Freeze feature extractor, keep the two dense heads trainable
for param in model.features.parameters():
    param.requires_grad = False
for param in model.fc1.parameters():
    param.requires_grad = True
for param in model.fc2.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

def evaluate(loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total if total else 0.0

print('Starting fine-tuning...')
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for imgs, labels, _ in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * imgs.size(0)
    epoch_loss /= len(train_loader.dataset)
    val_acc = evaluate(val_loader)
    print(f'Epoch {epoch:02d} | loss {epoch_loss:.4f} | val-acc {val_acc*100:.2f}%')

print('Fine-tuning finished!')

# Save new weights – compatible with load_anemia_model
out_path = 'M_Anemia_finetuned.h5'
torch.save(model.state_dict(), out_path)
print(f'Fine-tuned weights saved to {out_path}')
