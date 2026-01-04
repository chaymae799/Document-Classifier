"""
TRAINING SIMPLIFIÉ - CV ONLY (pas de Gabarits)
Pour debug rapide pendant le développement
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
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLASSES = ['facture_eau', 'facture_electricite', 'releve_bancaire']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

class FastDataset(Dataset):
    """Dataset simplifié - CV ONLY sans Gabarits"""
    
    def __init__(self, data_dir, split='train'):
        self.data_dir = Path(data_dir)
        self.split = split
        self.images = []
        self.labels = []
        
        # Charger les images
        for cls_idx, cls_name in enumerate(CLASSES):
            cls_dir = self.data_dir / split / cls_name
            if cls_dir.exists():
                for img_path in cls_dir.glob('*.jpg'):
                    self.images.append(str(img_path))
                    self.labels.append(cls_idx)
                for img_path in cls_dir.glob('*.png'):
                    self.images.append(str(img_path))
                    self.labels.append(cls_idx)
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            from PIL import Image
            img = Image.fromarray(img)
            img = self.transform(img)
        except:
            img = torch.zeros(3, 224, 224)
        
        return img, label

class HybridResNet50(nn.Module):
    """ResNet50 avec fusion simple (CV only pour ce test)"""
    
    def __init__(self, num_classes=3):
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': 100*correct/total})
    
    return total_loss / len(train_loader), 100 * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return total_loss / len(val_loader), 100 * correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data_augmented')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Datasets
    logger.info(f"Loading dataset from {args.data_dir}...")
    train_dataset = FastDataset(args.data_dir, split='train')
    val_dataset = FastDataset(args.data_dir, split='val')
    
    logger.info(f"✓ Train: {len(train_dataset)} images")
    logger.info(f"✓ Val: {len(val_dataset)} images")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Model
    model = HybridResNet50(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=2, verbose=True)
    
    best_val_acc = 0
    
    # Training loop
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = Path('models/cv') / f'model_epoch_{epoch+1:02d}_valacc_{val_acc:.4f}.pth'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.resnet.state_dict(), checkpoint_path)
            logger.info(f"  ✓ Checkpoint: {checkpoint_path}")
    
    # Save final model
    output_path = Path('models/cv/hybrid_resnet50.pth')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.resnet.state_dict(), output_path)
    logger.info(f"✓ Modèle sauvegardé: {output_path}")

if __name__ == '__main__':
    main()
