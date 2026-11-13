"""
Fine‑tune the existing anemia detection model using the small labelled
datasets under `test_dataset/` (Thalassemia) and `test_iron/`
(Iron Deficiency Anemia). The script does **not** download any external
weights – it loads the already‑trained `M_Anemia.h5` checkpoint and
continues training with heavy data augmentation.

The workflow:
  1. Collect image paths and assign a class based on the folder they
     belong to (Thalassemia or Iron Deficiency).
  2. Apply strong random augmentations to compensate for the tiny
     training set (38 images total).
  3. Train the `AnemiaVGGModel` (VGG‑like architecture) for a few
     epochs, first freezing the convolutional blocks, then un‑freezing
     everything.
  4. Save the new weights to `M_Anemia_finetuned.h5` and clear the
     model cache so the Flask app will load the updated model.

You can run the script with:
    python train_with_labels.py
"""

import argparse

parser = argparse.ArgumentParser(description='Fine‑tune anemia model on selected folders')
parser.add_argument('--folders', type=str, default='test_dataset,test_iron',
                    help='Comma‑separated list of folders to train on (e.g., "sickle cell")')
args = parser.parse_args()
folder_list = [f.strip() for f in args.folders.split(',') if f.strip()]

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 4
EPOCHS = 30               # Small data – a few epochs are enough
LR_PHASE1 = 1e-4           # When only top layers train
LR_PHASE2 = 5e-5           # Fine‑tune whole network
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------
# Helper: map folder name to target class index in CLASSES
# ---------------------------------------------------------------------
FOLDER_CLASS_MAP = {
    'test_dataset': 'Thalassemia',
    'test_iron':    'Iron Deficiency Anemia',
    'sickle cell':  'Sickle Cell Anemia',
}

def get_class_idx(folder_name: str) -> int:
    """Return the integer index for the disease represented by *folder_name*."""
    disease = FOLDER_CLASS_MAP.get(folder_name)
    if disease is None:
        raise ValueError(f'Unknown training folder: {folder_name}')
    return CLASSES.index(disease)

# ---------------------------------------------------------------------
# Dataset definition
# ---------------------------------------------------------------------
class SimpleAnemiaDataset(Dataset):
    """Loads images from a list and returns (tensor, class_idx)."""
    def __init__(self, image_paths, class_idxs, transform=None):
        self.image_paths = image_paths
        self.class_idxs = class_idxs
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.class_idxs[idx]

# ---------------------------------------------------------------------
# Gather data
# ---------------------------------------------------------------------
image_paths = []
class_idxs = []

for folder in ['test_dataset', 'test_iron', 'sickle cell']:
    base = Path(folder)
    cls_idx = get_class_idx(folder)
    for img_file in sorted(base.glob('*')):
        if img_file.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}:
            image_paths.append(str(img_file))
            class_idxs.append(cls_idx)

print('Collected training data:')
print(f'  Total images : {len(image_paths)}')
print(f'  Thalassemia  : {class_idxs.count(CLASSES.index("Thalassemia"))}')
print(f'  Iron Def.    : {class_idxs.count(CLASSES.index("Iron Deficiency Anemia"))}')

# ---------------------------------------------------------------------
# Data augmentation (heavier than the original script)
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Split into train/validation (20% validation)
# ---------------------------------------------------------------------
full_dataset = SimpleAnemiaDataset(image_paths, class_idxs, transform=train_transform)
val_dataset = SimpleAnemiaDataset(image_paths, class_idxs, transform=val_transform)

val_len = int(0.2 * len(full_dataset))
train_len = len(full_dataset) - val_len
train_set, _ = random_split(full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(SEED))
# Use the *same* images for validation but without augmentation
val_set, _ = random_split(val_dataset, [val_len, train_len], generator=torch.Generator().manual_seed(SEED))

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ---------------------------------------------------------------------
# Load the original model weights (acts as a good initialization)
# ---------------------------------------------------------------------
model = load_anemia_model(h5_path='M_Anemia.h5').to(DEVICE)
print('Loaded M_Anemia.h5 as starting checkpoint')

# ---------------------------------------------------------------------
# Phase 1 – train only the two dense heads (freeze conv blocks)
# ---------------------------------------------------------------------
for param in model.features.parameters():
    param.requires_grad = False
# FC layers remain trainable (they were already trainable)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_PHASE1)

best_acc = 0.0
print('\n--- Phase 1: Training heads only (epochs 1-10) ---')
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
    print(f'Epoch {epoch:02d} | loss {epoch_loss:.4f} | val-acc {acc*100:.1f}%')
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'M_Anemia_finetuned.h5')

# ---------------------------------------------------------------------
# Phase 2 – unfreeze everything and fine‑tune with a lower LR
# ---------------------------------------------------------------------
for param in model.features.parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam(model.parameters(), lr=LR_PHASE2)
print('\n--- Phase 2: Full fine-tuning (epochs 11-30) ---')
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
    print(f'Epoch {epoch:02d} | loss {epoch_loss:.4f} | val-acc {acc*100:.1f}%')
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'M_Anemia_finetuned.h5')

print('\nTraining complete! Best validation accuracy: {:.1f}%'.format(best_acc*100))
print('Fine-tuned weights saved to M_Anemia_finetuned.h5')

# ---------------------------------------------------------------------
# Clear cached model so the Flask app loads the new checkpoint next time.
# ---------------------------------------------------------------------
reset_model_cache()
print('Model cache cleared.')
