"""
MODULE 3 COMPLET: FINE-TUNING RÉSEAU HYBRIDE (CV + GABARITS)
Selon les demandes du projet:
- Fine-tune ResNet50 backbone
- Fusion avec features gabarits (20-30 features structurelles)
- Classification 5 classes avec validation par règles métier

Usage:
    python backend/train.py --data-dir data --output models/cv/hybrid_resnet50.pth --epochs 20 --batch-size 8
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import json
import sys
from datetime import datetime
import logging

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Classes
CLASSES = ['facture_eau', 'facture_electricite', 'piece_identite', 'releve_bancaire', 'document_employeur']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}


class GabaritsExtractor:
    """Extrait les 30 features structurelles pour la branche gabarits"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        # OCR text directory (optional). If present, OCR-derived features will be appended.
        self.ocr_dir = Path(__file__).parent.parent / 'data' / 'ocr_text'
    
    def extract(self, image_path):
        """Extrait 36 features structurelles d'une image (30 vision + 6 OCR-derived)"""
        from PIL import Image as PILImage
        try:
            pil_img = PILImage.open(image_path).convert('RGB')
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            return np.zeros(36, dtype=np.float32)
        
        if img is None:
            return np.zeros(36, dtype=np.float32)
        
        h, w = img.shape[:2]
        features = []
        
        # 1-2: Aspect ratio + ratio carré
        aspect_ratio = h / w if w > 0 else 0
        features.append(min(aspect_ratio, 2.0))  # Capper à 2.0
        features.append(abs(1 - aspect_ratio))  # Deviation du carré
        
        # 3-5: Détection de photo (visage)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        features.append(1.0 if len(faces) > 0 else 0.0)
        features.append(float(len(faces)) / 10.0)  # Nombre de visages normalisé
        
        # Confiance photo (taille/position)
        photo_conf = 0.0
        for (x, y, fw, fh) in faces:
            size_ratio = (fw * fh) / (w * h)
            photo_conf = max(photo_conf, min(size_ratio * 20, 1.0))
        features.append(photo_conf)
        
        # 6-7: Densité de texte
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        text_density = cv2.countNonZero(binary) / (h * w)
        features.append(text_density)
        features.append(1 - text_density)  # Densité inverse
        
        # 8-9: Détection de structure tabulaire
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        h_lines = v_lines = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle < 10 or angle > 170:
                    h_lines += 1
                elif 80 < angle < 100:
                    v_lines += 1
        
        features.append(min(h_lines / 20.0, 1.0))  # Lignes horizontales normalisées
        features.append(min(v_lines / 20.0, 1.0))  # Lignes verticales normalisées
        
        # 10: Confiance structure tabulaire
        table_conf = 1.0 if (h_lines >= 3 and v_lines >= 2) else 0.0
        features.append(table_conf)
        
        # 11-14: Analyse de couleurs (HSV)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Bleu
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([130, 255, 255])
        blue_ratio = cv2.countNonZero(cv2.inRange(hsv, blue_lower, blue_upper)) / (h * w)
        features.append(blue_ratio)
        
        # Rouge
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 50, 50])
        red_upper2 = np.array([180, 255, 255])
        red_mask = cv2.bitwise_or(cv2.inRange(hsv, red_lower1, red_upper1),
                                  cv2.inRange(hsv, red_lower2, red_upper2))
        red_ratio = cv2.countNonZero(red_mask) / (h * w)
        features.append(red_ratio)
        
        # Vert
        green_lower = np.array([40, 50, 50])
        green_upper = np.array([80, 255, 255])
        green_ratio = cv2.countNonZero(cv2.inRange(hsv, green_lower, green_upper)) / (h * w)
        features.append(green_ratio)
        
        # Couleur dominant
        dominant_color = max([blue_ratio, red_ratio, green_ratio])
        features.append(dominant_color)
        
        # 15-17: Détection de logo (en haut)
        top_region = img[:int(h * 0.2), :]
        gray_top = cv2.cvtColor(top_region, cv2.COLOR_BGR2GRAY)
        _, binary_top = cv2.threshold(gray_top, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary_top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        logo_candidates = sum(1 for c in contours if 1000 < cv2.contourArea(c) < 50000)
        features.append(1.0 if logo_candidates > 0 else 0.0)
        features.append(float(logo_candidates) / 5.0)
        
        # Densité du logo
        logo_density = cv2.countNonZero(binary_top) / (top_region.shape[0] * top_region.shape[1])
        features.append(logo_density)
        
        # 18-20: Détection de graphiques
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=30,
                                   minRadius=20, maxRadius=100)
        has_circles = 1.0 if circles is not None else 0.0
        features.append(has_circles)
        
        curved_contours = sum(1 for c in contours if len(c) > 50)
        features.append(min(curved_contours / 5.0, 1.0))
        
        # Confiance graphiques
        graph_conf = (has_circles * 0.5) + (min(curved_contours / 5.0, 1.0) * 0.5)
        features.append(graph_conf)
        
        # 21-23: Analyse layout (densité par région)
        top_third = gray[:int(h * 0.33), :]
        mid_third = gray[int(h * 0.33):int(h * 0.66), :]
        bot_third = gray[int(h * 0.66):, :]
        
        top_density = np.sum(top_third < 200) / max(top_third.size, 1)
        mid_density = np.sum(mid_third < 200) / max(mid_third.size, 1)
        bot_density = np.sum(bot_third < 200) / max(bot_third.size, 1)
        
        features.append(top_density)
        features.append(mid_density)
        features.append(bot_density)
        
        # 24: Équilibre vertical
        balance = 1.0 - abs(top_density - bot_density)
        features.append(balance)
        
        # 25-27: Détection de signature (bas)
        bottom_third = img[int(h * 0.66):, :]
        gray_bottom = cv2.cvtColor(bottom_third, cv2.COLOR_BGR2GRAY)
        edges_bottom = cv2.Canny(gray_bottom, 30, 100)
        contours_bottom, _ = cv2.findContours(edges_bottom, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        sig_candidates = 0
        for c in contours_bottom:
            x, y, w_cont, h_cont = cv2.boundingRect(c)
            if h_cont > 0:
                ar = w_cont / h_cont
                if 2 < ar < 8 and w_cont > 50:
                    sig_candidates += 1
        
        features.append(1.0 if sig_candidates > 0 else 0.0)
        features.append(float(sig_candidates) / 5.0)
        features.append(bot_density)  # Densité signature
        
        # 28-30: Contraste et netteté
        contrast = gray.std()
        features.append(min(contrast / 80.0, 1.0))
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        features.append(min(sharpness / 500.0, 1.0))
        
        # Qualité générale
        quality = (min(contrast / 80.0, 1.0) + min(sharpness / 500.0, 1.0)) / 2
        features.append(quality)
        
        # ----- OCR-derived features (6 features) -----
        # Word count, avg_word_len, digit_ratio, has_facture_keyword, has_montant_keyword, has_iban
        try:
            ocr_file = self.ocr_dir / (Path(image_path).stem + '.txt')
            if ocr_file.exists():
                text = ocr_file.read_text(encoding='utf-8')
                words = [w for w in text.split() if len(w) > 0]
                word_count = len(words)
                avg_word_len = sum(len(w) for w in words) / word_count if word_count > 0 else 0.0
                chars = list(text)
                digit_ratio = sum(c.isdigit() for c in chars) / max(len(chars), 1)
                lower = text.lower()
                has_facture = 1.0 if ('facture' in lower or 'factures' in lower) else 0.0
                has_montant = 1.0 if ('montant' in lower or 'total' in lower or 'tds' in lower) else 0.0
                has_iban = 1.0 if ('iban' in lower or 'bank' in lower or 'compte' in lower) else 0.0
            else:
                word_count = 0.0
                avg_word_len = 0.0
                digit_ratio = 0.0
                has_facture = 0.0
                has_montant = 0.0
                has_iban = 0.0
        except Exception:
            word_count = 0.0
            avg_word_len = 0.0
            digit_ratio = 0.0
            has_facture = 0.0
            has_montant = 0.0
            has_iban = 0.0

        features.append(min(word_count / 500.0, 1.0))
        features.append(min(avg_word_len / 10.0, 1.0))
        features.append(digit_ratio)
        features.append(has_facture)
        features.append(has_montant)
        features.append(has_iban)

        return np.array(features, dtype=np.float32)


class DocumentDataset(Dataset):
    """Dataset pour images de documents avec labels et features gabarits"""
    
    def __init__(self, data_dir, split='train', gabarit_extractor=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.gabarit_extractor = gabarit_extractor or GabaritsExtractor()
        
        # Transforms with stronger augmentations for training
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomRotation(8),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.02),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        
        # Charger les images
        self.images = []
        self.labels = []
        self.gabarit_features = []
        
        split_dir = self.data_dir / split
        for class_name in CLASSES:
            class_dir = split_dir / class_name
            if class_dir.exists():
                for img_path in list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')):
                    self.images.append(img_path)
                    self.labels.append(CLASS_TO_IDX[class_name])
                    
                    # Extraire features gabarits
                    gab_feat = self.gabarit_extractor.extract(img_path)
                    self.gabarit_features.append(gab_feat)
        
        logger.info(f"Dataset {split}: {len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        gab_feat = self.gabarit_features[idx]
        
        # Force PIL for all image loading (handles unicode paths)
        from PIL import Image as PILImage
        try:
            img = PILImage.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Cannot load {img_path}: {e}, returning blank image")
            img = PILImage.new('RGB', (224, 224), color=(128, 128, 128))
        
        img = self.transforms(img)
        
        return {
            'image': img,
            'gabarit': torch.from_numpy(gab_feat),
            'label': torch.tensor(label, dtype=torch.long)
        }


class HybridResNet50(nn.Module):
    """Modèle hybride: ResNet50 (image) + Dense (gabarits) + Fusion"""
    
    def __init__(self, num_classes=5):
        super().__init__()
        
        # Branche CV: ResNet50 pretrained
        self.resnet50 = resnet50(pretrained=True)
        # Remplacer la dernière couche
        self.resnet50.fc = nn.Identity()  # Retirer la classification
        
        # Branche gabarits: réseau dense (now accepts 36 features including OCR-derived)
        self.gabarit_net = nn.Sequential(
            nn.Linear(36, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Fusion: concaténer ResNet (2048) + Gabarits (64) = 2112
        self.fusion = nn.Sequential(
            nn.Linear(2048 + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, image, gabarit):
        # Branche CV
        cv_features = self.resnet50(image)  # (batch, 2048)
        
        # Branche gabarits
        gab_features = self.gabarit_net(gabarit)  # (batch, 64)
        
        # Fusion
        combined = torch.cat([cv_features, gab_features], dim=1)  # (batch, 2112)
        logits = self.fusion(combined)
        
        return logits


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Entraîne un epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        images = batch['image'].to(device)
        gabarits = batch['gabarit'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images, gabarits)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': total_loss / (pbar.n + 1), 'acc': correct / total})
    
    return total_loss / len(train_loader), correct / total


def eval_epoch(model, val_loader, criterion, device):
    """Évalue un epoch"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            images = batch['image'].to(device)
            gabarits = batch['gabarit'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images, gabarits)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(val_loader), correct / total


def compute_class_weights(dataset):
    # Compute inverse frequency class weights
    from collections import Counter
    labels = [int(x) for x in dataset.labels]
    cnt = Counter(labels)
    total = sum(cnt.values())
    weights = []
    for i in range(len(CLASSES)):
        weights.append(total / (cnt.get(i, 1)))
    # Normalize
    w = np.array(weights, dtype=np.float32)
    w = w / w.sum() * len(CLASSES)
    return torch.tensor(w, dtype=torch.float)


def main():
    parser = argparse.ArgumentParser(description='Fine-tune ResNet50 + Gabarits hybride')
    parser.add_argument('--data-dir', default='data', help='Chemin vers le dataset')
    parser.add_argument('--output', default='backend/models/cv/hybrid_resnet50.pth',
                       help='Chemin de sauvegarde du modèle')
    parser.add_argument('--epochs', type=int, default=15, help='Nombre d\'epochs (15 pour démo rapide, 50-80 pour production)')
    parser.add_argument('--batch-size', type=int, default=16, help='Taille des batches')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    logger.info(f"Device: {device}")
    
    # Datasets
    gabarit_extractor = GabaritsExtractor()
    train_dataset = DocumentDataset(args.data_dir, split='train', gabarit_extractor=gabarit_extractor)
    val_dataset = DocumentDataset(args.data_dir, split='val', gabarit_extractor=gabarit_extractor)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Modèle
    model = HybridResNet50(num_classes=len(CLASSES)).to(device)

    # Class weights to mitigate imbalance
    class_weights = compute_class_weights(train_dataset).to(device)
    logger.info(f"Class weights: {class_weights}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimiseur avec learning rate réduit pour fine-tuning
    optimizer = optim.Adam([
        {'params': model.resnet50.parameters(), 'lr': args.lr * 0.05},  # Backbone: smaller LR
        {'params': model.gabarit_net.parameters(), 'lr': args.lr},      # Gabarits: LR normal
        {'params': model.fusion.parameters(), 'lr': args.lr}            # Fusion: LR normal
    ], weight_decay=1e-4)

    # Scheduler and early stopping
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                      patience=2, verbose=True)
    early_stop_patience = 5  # Arrêt si pas d'amélioration en 5 epochs
    epochs_no_improve = 0

    # Checkpoint dir
    ckpt_dir = Path(args.output).parent / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Training
    best_val_acc = 0.0
    results = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    logger.info(f"\nDémarrage du fine-tuning ({args.epochs} epochs)...")
    logger.info(f"Train: {len(train_dataset)} images | Val: {len(val_dataset)} images\n")
    
    for epoch in range(args.epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        logger.info(f"{'='*60}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)
        
        logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Sauvegarder le meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_to_idx': CLASS_TO_IDX,
                'idx_to_class': IDX_TO_CLASS,
                'epoch': epoch,
                'best_val_acc': val_acc
            }, output_path)
            logger.info(f"✓ Modèle sauvegardé: {output_path} (Acc: {val_acc:.4f})")
            # also save checkpoint
            torch.save(model.state_dict(), ckpt_dir / f'model_epoch_{epoch+1:02d}_valacc_{val_acc:.4f}.pth')
        else:
            epochs_no_improve += 1

        scheduler.step(val_acc)

        # Early stopping
        if epochs_no_improve >= early_stop_patience:
            logger.info(f"Early stopping: no improvement for {early_stop_patience} epochs")
            break
    
    # Sauvegarder les résultats
    results_path = Path(args.output).parent / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Fine-tuning terminé!")
    logger.info(f"Meilleure val accuracy: {best_val_acc:.4f}")
    logger.info(f"Résultats sauvegardés: {results_path}")
    logger.info(f"{'='*60}\n")


if __name__ == '__main__':
    main()
