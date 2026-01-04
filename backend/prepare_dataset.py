"""
Direct dataset preparation: Raw images → Clean → Train/Val splits
Handles unicode filenames, applies preprocessing, and organizes data.
"""
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
import shutil
import unicodedata
from collections import defaultdict

RAW_DIR = Path('data/raw_images')
TRAIN_DIR = Path('data/train')
VAL_DIR = Path('data/val')

CATEGORIES = {
    'facture_eau': 'facture_eau',
    'facture_electricite': 'facture_electricite',
    'releve_bancaire': 'releve_bancaire',
    'piece_identite': 'piece_identite',
    'document_employeur': 'document_employeur',
}

TRAIN_RATIO = 0.8


def normalize(text: str) -> str:
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode().lower()


def extract_category(filename):
    """Robust category extraction with keyword priority."""
    norm = normalize(filename)

    if ('releve' in norm) or ('banc' in norm):
        return 'releve_bancaire'
    if ('electricite' in norm) or ('electr' in norm):
        return 'facture_electricite'
    if 'eau' in norm:
        return 'facture_eau'
    if ('piece' in norm) or ('identite' in norm) or ('identity' in norm):
        return 'piece_identite'
    if ('employeur' in norm) or ('employer' in norm):
        return 'document_employeur'
    # Fallback
    return 'facture_electricite'


def clean_image(img):
    """Apply preprocessing: denoise, contrast, sharpen."""
    # Bilateral denoise
    img = cv2.bilateralFilter(img, 9, 75, 75)
    
    # CLAHE contrast enhancement
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Additional denoising
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    
    # Sharpening
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) / 1.0
    img_enhanced = cv2.filter2D(img, -1, kernel)
    
    # Normalize
    img_enhanced = cv2.normalize(img_enhanced, None, 0, 255, cv2.NORM_MINMAX)
    
    return img_enhanced


def prepare_dataset():
    """Main pipeline: load → clean → split → save."""
    print("=" * 70)
    print("DATASET PREPARATION: RAW → CLEAN → TRAIN/VAL")
    print("=" * 70)
    
    # Reset train/val to avoid stale/misplaced files
    for d in [TRAIN_DIR, VAL_DIR]:
        if d.exists():
            shutil.rmtree(d)

    # Create directories
    for cat in CATEGORIES.values():
        (TRAIN_DIR / cat).mkdir(parents=True, exist_ok=True)
        (VAL_DIR / cat).mkdir(parents=True, exist_ok=True)
    
    # Collect images by category
    images_by_cat = defaultdict(list)
    
    print(f"\n📂 Loading {len(list(RAW_DIR.glob('*.jpg')))} raw images...\n")
    
    for raw_img in sorted(RAW_DIR.glob('*.jpg')):
        cat = extract_category(raw_img.name)
        
        # Load with cv2, fallback to PIL
        img = cv2.imread(str(raw_img))
        if img is None:
            try:
                pil_img = Image.open(raw_img).convert('RGB')
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"✗ Cannot load {raw_img.name}: {e}")
                continue
        
        # Clean
        img_clean = clean_image(img)
        
        # Save to appropriate category folder in train_dir (temp location)
        temp_dir = TRAIN_DIR / cat
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / raw_img.name
        
        try:
            cv2.imwrite(str(temp_path), img_clean)
            images_by_cat[cat].append(temp_path)
            print(f"✓ {raw_img.name} → {cat}")
        except Exception as e:
            print(f"✗ Cannot save {raw_img.name}: {e}")
    
    print(f"\n✅ Cleaned and saved {sum(len(v) for v in images_by_cat.values())} images\n")
    
    # Now split train/val
    print("📋 Creating train/val splits (80/20)...\n")
    
    train_count = 0
    val_count = 0
    
    for cat in CATEGORIES.values():
        cat_train_dir = TRAIN_DIR / cat
        cat_val_dir = VAL_DIR / cat
        
        images = sorted([f for f in cat_train_dir.glob('*.jpg')])
        if not images:
            continue
        
        split_idx = int(len(images) * TRAIN_RATIO)
        
        # Keep first 80% in train
        # Move last 20% to val
        for i, img in enumerate(images):
            if i >= split_idx:
                try:
                    shutil.move(str(img), str(cat_val_dir / img.name))
                    val_count += 1
                except Exception as e:
                    print(f"✗ Cannot move {img.name} to val: {e}")
            else:
                train_count += 1
        
        print(f"   {cat}: {split_idx} train, {len(images) - split_idx} val")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ DATASET READY")
    print("=" * 70)
    print(f"\n📊 Final splits:")
    print(f"   Train: {train_count} images")
    print(f"   Val: {val_count} images")
    print(f"   Total: {train_count + val_count} images")
    
    # Verify
    t = len(list(TRAIN_DIR.rglob('*.jpg')))
    v = len(list(VAL_DIR.rglob('*.jpg')))
    print(f"\n✓ Verified: {t} train + {v} val = {t + v} total")
    
    return (t > 0) and (v > 0)


if __name__ == '__main__':
    success = prepare_dataset()
    exit(0 if success else 1)
