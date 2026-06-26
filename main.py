import os
os.system("pip install -q timm albumentations")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import timm

# --- 1. CONFIGURATION ---
class CFG:
    OOD_DATA_DIR = Path("/kaggle/input/datasets/nesine/dataset-test/dataset_test_Rabie_cleaned/tomato")
    WEIGHTS_PATH = Path("/kaggle/input/datasets/nesine/tomato-model-ensemble-learning-v1/ensemble_best_weights.pt")
    IMG_SIZE = 256
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    NUM_CLASSES = 7
    BACKBONES = ["convnext_nano.in12k_ft_in1k", "efficientnet_b0", "resnet34d"]
    CLASS_NAMES = [
        "Bacterial spot", "Early blight", "Healthy", "Late blight",
        "Leaf mold", "Septoria leaf spot", "Yellow leaf curl virus"
    ]

# --- 2. CORRECTION DU CORRESPONDANCE DOSSIER -> CLASSE ---
OOD_TO_TRAINED = {
    "bacterial_spot": "Bacterial spot",
    "early_blight": "Early blight",
    "healthy": "Healthy",
    "late_blight": "Late blight",
    "leaf_mold": "Leaf mold",
    "septoria_leaf_spot": "Septoria leaf spot",
    "yellow_leaf_curl_virus": "Yellow leaf curl virus"
    # Note : "mosaic_virus", "spider_mites" et "verticillium_wilt" ne font pas partie 
    # des 7 classes de CFG.CLASS_NAMES, ils seront ignorés lors de la découverte.
}

label_to_idx = {name: i for i, name in enumerate(CFG.CLASS_NAMES)}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 3. CHARGEMENT DES IMAGES DE TEST ---
def discover_images(data_dir):
    records = []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    if not data_dir.exists():
        return records

    for folder in sorted(data_dir.iterdir()):
        if folder.is_dir() and folder.name in OOD_TO_TRAINED:
            class_name = OOD_TO_TRAINED[folder.name]
            for img_path in folder.iterdir():
                if img_path.suffix.lower() in valid_exts:
                    records.append((str(img_path), class_name))
    return records

ood_records = discover_images(CFG.OOD_DATA_DIR)
print(f"Nombre d'images de test OOD trouvées : {len(ood_records)}")

if len(ood_records) == 0:
    raise ValueError("Le dataset est toujours vide. Vérifiez le chemin : " + str(CFG.OOD_DATA_DIR))

ood_paths = [r[0] for r in ood_records]
ood_targets = [label_to_idx[r[1]] for r in ood_records]

# --- 4. DATASET & TRANSFORMS ---
test_transform = A.Compose([
    A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

class TomatoDataset(Dataset):
    def __init__(self, image_paths, targets, transform=None):
        self.image_paths = image_paths
        self.targets = targets
        self.transform = transform
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, self.targets[idx]

ood_dataset = TomatoDataset(ood_paths, ood_targets, transform=test_transform)
ood_loader = DataLoader(ood_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)

# --- 5. ARCHITECTURE ENSEMBLE ---
class TomatoEnsembleModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.model1 = timm.create_model(CFG.BACKBONES[0], pretrained=False, num_classes=num_classes)
        self.model2 = timm.create_model(CFG.BACKBONES[1], pretrained=False, num_classes=num_classes)
        self.model3 = timm.create_model(CFG.BACKBONES[2], pretrained=False, num_classes=num_classes)
    def forward(self, x):
        return self.model1(x), self.model2(x), self.model3(x)

# Initialisation et chargement des poids appris
model = TomatoEnsembleModel(num_classes=CFG.NUM_CLASSES).to(device)
model.load_state_dict(torch.load(CFG.WEIGHTS_PATH, map_location=device))
model.eval()
print("Poids du Voting Ensemble chargés avec succès !")

# --- 6. EVALUATION (SOFT VOTING) ---
all_preds, all_targets = [], []

with torch.no_grad():
    for images, targets in ood_loader:
        images = images.to(device)
        out1, out2, out3 = model(images)
        
        prob1 = F.softmax(out1, dim=1)
        prob2 = F.softmax(out2, dim=1)
        prob3 = F.softmax(out3, dim=1)
        
        ensemble_probs = (prob1 + prob2 + prob3) / 3.0
        preds = ensemble_probs.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.numpy())

# --- 7. AFFICHAGE DES RÉSULTATS ---
all_targets = np.array(all_targets)
all_preds = np.array(all_preds)

print("\n================== RAPPORT DE PERFORMANCES OOD ==================")
print(f"OOD Accuracy: {100 * accuracy_score(all_targets, all_preds):.2f}%")
print(classification_report(all_targets, all_preds, target_names=CFG.CLASS_NAMES, zero_division=0))
