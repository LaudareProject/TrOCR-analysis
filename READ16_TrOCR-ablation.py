#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ===================================================================
# 🎯 SELECT WHICH CONFIGURATION TO RUN
# ===================================================================
SELECTED_CONFIG = {
    "name": "enc_0_dec_0_no_aug",
    "freeze_encoder": 0,   # 0 = fully trainable encoder
    "freeze_decoder": 0,   # 0 = fully trainable decoder
}

SEED = 42
USE_CLAHE  = True   # NO CLAHE — ablation test
USE_AUG    = False    # Augmentation applied to training set


# In[2]:


# ===================================================================
# CELL 2: Imports and Configuration
# ===================================================================

import os
import json
import random
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import shutil
import time

# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ===================================================================
# 📁 PATHS — adjust if your Drive folder name differs
# ===================================================================
# READ16 dataset root in Drive: should contain training/, validation/, testing/
# Each subfolder should have: pagexml/ and images/ (or Images/)
DRIVE_READ16 = Path("./READ16-data/PublicData")

# Local fast-storage copy (SSD on Colab runtime)
LOCAL_READ16  = Path("./read16_local")

# Output directory for checkpoints & results
OUTPUT_BASE = Path("./trocr_ablation_results")

OUTPUT_BASE.mkdir(exist_ok=True)
LOCAL_READ16.mkdir(exist_ok=True)

# ===================================================================
# ONE-CYCLE LR SCHEDULER HYPERPARAMETERS (same as Cortonese)
# ===================================================================
USE_ONECYCLE           = True
ONECYCLE_MAX_LR        = 5.5e-6
ONECYCLE_PCT_START     = 0.1
ONECYCLE_BASE_MOMENTUM = 0.85
ONECYCLE_MAX_MOMENTUM  = 0.95
ONECYCLE_INITIAL_LR    = 1e-9
ONECYCLE_FINAL_DIV_FACTOR = 2.2e4

# Training hyperparameters
NUM_EPOCHS  = 50
TRAIN_BATCH = 8
EVAL_BATCH  = 16
GRAD_ACCUM  = 2
LR = ONECYCLE_MAX_LR if USE_ONECYCLE else 3e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable CUDA error checking & clear cache
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.cuda.empty_cache()

print("=" * 80)
print("🎯 SELECTED CONFIGURATION")
print("=" * 80)
print(f"Configuration  : {SELECTED_CONFIG['name']}")
print(f"Encoder frozen : {SELECTED_CONFIG['freeze_encoder']}/12 layers")
print(f"Decoder frozen : {SELECTED_CONFIG['freeze_decoder']}/6 layers")
print(f"\n🚀 Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"\n📊 Training Configuration:")
print(f"   Epochs        : {NUM_EPOCHS}")
print(f"   One-Cycle LR  : {USE_ONECYCLE}")
if USE_ONECYCLE:
    print(f"   Max LR        : {ONECYCLE_MAX_LR}")
    print(f"   Initial LR    : {ONECYCLE_INITIAL_LR}")
    print(f"   Warmup frac   : {ONECYCLE_PCT_START}")
print(f"   Batch size    : {TRAIN_BATCH}  (grad accum {GRAD_ACCUM})")
print(f"   CLAHE         : {USE_CLAHE}")
print(f"   Augmentation  : {USE_AUG}")
print("=" * 80)
print("✅ Configuration complete!")


# In[3]:


# ===================================================================
# CELL 3: Copy READ16 Dataset to Local Storage (faster I/O)
# ===================================================================
# Expected folder structure inside DRIVE_READ16:
#   training/
#       pagexml/   ← PAGE XML annotation files
#       images/    ← page images (jpg / JPG / png)
#   validation/
#       pagexml/
#       images/
#   testing/
#       pagexml/
#       images/

print("→️ Copying READ16 dataset to local storage...")
print(f"   Source     : {DRIVE_READ16}")
print(f"   Destination: {LOCAL_READ16}")

if not DRIVE_READ16.exists():
    raise FileNotFoundError(
        f"❌ Dataset not found at {DRIVE_READ16}\n"
        "   Please check your Google Drive path."
    )

# Check if copy already exists and looks complete (has subdirs)
local_subdirs = [p for p in LOCAL_READ16.iterdir() if p.is_dir()] if LOCAL_READ16.exists() else []
if not local_subdirs:
    print("   Copying now (this may take a few minutes)...")
    shutil.copytree(str(DRIVE_READ16), str(LOCAL_READ16), dirs_exist_ok=True)
    print("✅ Copy complete!")
else:
    print(f"✅ Local copy already exists ({len(local_subdirs)} subfolders) — skipping copy.")

# ---------------------------------------------------------------
# Discover actual subfolder names (case-insensitive)
# ---------------------------------------------------------------
def find_subfolder(root: Path, candidates):
    """Return first existing subfolder matching any candidate name (case-insensitive)."""
    existing = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}
    for c in candidates:
        if c.lower() in existing:
            return existing[c.lower()]
    return None

def find_split_dirs(root: Path, split_names):
    """
    Find xml_dir and img_dir for a given split (training/validation/testing).
    Handles two structures:
      A) split_root/page/  + split_root/Images/       (Training, Validation)
      B) split_root/Subfolder/page/ + images in Subfolder root  (Testing/Test-ICFHR-2016)
    """
    split_root = find_subfolder(root, split_names)
    if split_root is None:
        raise FileNotFoundError(
            f"Could not find any of {split_names} under {root}. "
            f"Available dirs: {[p.name for p in root.iterdir() if p.is_dir()]}"
        )

    # Try direct structure first: split_root/page/ and split_root/Images/
    xml_dir = find_subfolder(split_root, ["pagexml", "page", "page_xml", "xml"])
    img_dir = find_subfolder(split_root, ["images", "image", "imgs", "img"])

    # If not found directly, check one level deeper (e.g. Testing/Test-ICFHR-2016/)
    if xml_dir is None or img_dir is None:
        subdirs = [p for p in split_root.iterdir() if p.is_dir()]
        if len(subdirs) == 1:
            nested = subdirs[0]
            xml_dir = find_subfolder(nested, ["pagexml", "page", "page_xml", "xml"])
            img_dir = find_subfolder(nested, ["images", "image", "imgs", "img"])
            # Handle case where images live directly in nested root (no images/ subfolder)
            if img_dir is None:
                if any(f.is_file() and f.suffix.lower() in ['.jpg', '.JPG', '.png', '.PNG', '.tif', '.TIF']
                       for f in nested.iterdir()):
                    img_dir = nested  # images are directly in the nested folder root

    if xml_dir is None or img_dir is None:
        raise FileNotFoundError(
            f"Could not find page XMLs and images inside {split_root}.\n"
            f"  xml_dir found: {xml_dir}\n"
            f"  img_dir found: {img_dir}\n"
            f"  Contents: {[p.name for p in split_root.iterdir()]}"
        )
    return xml_dir, img_dir

# Discover paths for each split
TRAIN_XML_DIR,  TRAIN_IMG_DIR  = find_split_dirs(LOCAL_READ16, ["training",   "train"])
VAL_XML_DIR,    VAL_IMG_DIR    = find_split_dirs(LOCAL_READ16, ["validation", "val",  "valid"])
TEST_XML_DIR,   TEST_IMG_DIR   = find_split_dirs(LOCAL_READ16, ["testing",    "test"])

print("\n📂 Discovered dataset paths:")
for name, xd, id_ in [("Train", TRAIN_XML_DIR, TRAIN_IMG_DIR),
                       ("Val",   VAL_XML_DIR,   VAL_IMG_DIR),
                       ("Test",  TEST_XML_DIR,  TEST_IMG_DIR)]:
    n_xml = len(list(xd.glob("*.xml")))
    n_img = len(list(id_.glob("*.[jJpP][pPnN][gG]*")))
    print(f"   {name:5s} | XML dir : {xd}  ({n_xml} files)")
    print(f"         | Img dir : {id_}  ({n_img} files)")

DATASET_READY = True
print("\n✅ Dataset paths verified!")


# In[4]:


# ===================================================================
# CELL 4: PAGE XML Parser
# ===================================================================

def parse_page_xml(xml_file):
    """Extract text lines from PAGE XML format (handles multiple namespace versions)."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        NAMESPACES = {
            'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15',
            'page2010': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19',
        }

        # Find Page element across known namespaces
        page = None
        ns_uri = None
        for _, uri in NAMESPACES.items():
            page = root.find(f'.//{{{uri}}}Page')
            if page is not None:
                ns_uri = uri
                break
        if page is None:
            page = root.find('.//Page')   # no namespace fallback

        if page is None:
            return None, []

        image_filename = page.get('imageFilename')

        def _find_all(tag):
            if ns_uri:
                return root.findall(f'.//{{{ns_uri}}}{tag}')
            return root.findall(f'.//{tag}')

        def _find_one(elem, tag):
            if ns_uri:
                return elem.find(f'.//{{{ns_uri}}}{tag}')
            return elem.find(f'.//{tag}')

        text_lines = []
        for tl in _find_all('TextLine'):
            coords_elem  = _find_one(tl, 'Coords')
            unicode_elem = _find_one(tl, 'Unicode')

            coords_str = coords_elem.get('points') if coords_elem is not None else None
            text = unicode_elem.text if unicode_elem is not None and unicode_elem.text else ""

            if not text.strip() or not coords_str:
                continue
            try:
                points = coords_str.strip().split()
                coords = [list(map(int, map(float, p.split(',')))) for p in points]
                xs = [c[0] for c in coords]
                ys = [c[1] for c in coords]
                x, y = min(xs), min(ys)
                w, h = max(xs) - x, max(ys) - y
                if w > 0 and h > 0:
                    text_lines.append({'text': text.strip(), 'bbox': [x, y, w, h]})
            except Exception:
                continue

        return image_filename, text_lines

    except Exception as e:
        print(f"  ⚠️  Failed to parse {xml_file}: {e}")
        return None, []


def load_split(xml_dir: Path, img_dir: Path, split_name: str, id_offset: int = 0):
    """
    Load one dataset split (train / val / test) from PAGE XML files.
    Returns (images list, annotations list, next_id_offset).
    """
    print(f"\n📖 Loading {split_name} split...")
    images, annotations = [], []
    ann_id = id_offset

    xml_files = sorted(xml_dir.glob('*.xml'))
    print(f"   Found {len(xml_files)} PAGE XML files")

    skipped_no_img = 0
    skipped_no_text = 0

    for img_id, xml_file in enumerate(xml_files, start=id_offset):
        image_filename, text_lines = parse_page_xml(xml_file)

        if not image_filename or not text_lines:
            skipped_no_text += 1
            continue

        # Locate image file (try exact name, then stem + common extensions)
        img_path = img_dir / image_filename
        if not img_path.exists():
            stem = Path(image_filename).stem
            for ext in ['.jpg', '.JPG', '.png', '.PNG', '.tif', '.TIF']:
                candidate = img_dir / f"{stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    image_filename = img_path.name
                    break

        if not img_path.exists():
            skipped_no_img += 1
            continue

        img_cv = cv2.imread(str(img_path))
        if img_cv is None:
            skipped_no_img += 1
            continue

        h_img, w_img = img_cv.shape[:2]

        images.append({
            "id": img_id,
            "file_name": image_filename,
            "width": w_img,
            "height": h_img,
        })

        for line in text_lines:
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 6,
                "bbox": line['bbox'],
                "description": line['text'],
            })
            ann_id += 1

    print(f"   ✅ {split_name}: {len(images)} pages, {len(annotations)} text lines")
    if skipped_no_text or skipped_no_img:
        print(f"   ⚠️  Skipped: {skipped_no_text} (no text/xml), {skipped_no_img} (no image)")
    return images, annotations, ann_id

print("✅ PAGE XML parser functions defined!")


# In[5]:


# ===================================================================
# CELL 5: Load READ16 Dataset Using Pre-Defined Train/Val/Test Splits
# ===================================================================
# The READ16 dataset already comes with official training/validation/testing
# splits — we use those directly instead of doing a random re-split.

if not DATASET_READY:
    raise RuntimeError("Dataset not ready. Run Cell 3 first.")

# Load each split independently
train_images, train_anns, _off = load_split(TRAIN_XML_DIR, TRAIN_IMG_DIR, "Training",   id_offset=0)
val_images,   val_anns,   _off = load_split(VAL_XML_DIR,   VAL_IMG_DIR,   "Validation", id_offset=_off)
test_images,  test_anns,  _off = load_split(TEST_XML_DIR,  TEST_IMG_DIR,  "Testing",    id_offset=_off)

# Build a unified image_map (id → image info + which dir it came from)
image_dir_map = {}   # img_id → Path of the images directory
for img in train_images:
    image_dir_map[img['id']] = TRAIN_IMG_DIR
for img in val_images:
    image_dir_map[img['id']] = VAL_IMG_DIR
for img in test_images:
    image_dir_map[img['id']] = TEST_IMG_DIR

image_map = {}
for img in train_images + val_images + test_images:
    image_map[img['id']] = img

print(f"\n{'='*70}")
print("DATA SPLIT SUMMARY")
print(f"{'='*70}")
print(f"Train : {len(train_anns):>6,} text lines  ({len(train_images):>4} pages)")
print(f"Val   : {len(val_anns):>6,} text lines  ({len(val_images):>4} pages)")
print(f"Test  : {len(test_anns):>6,} text lines  ({len(test_images):>4} pages)")
print(f"Total : {len(train_anns)+len(val_anns)+len(test_anns):>6,} text lines")
print(f"{'='*70}")

# Sample texts
print("\nSample text lines (from training):")
for i, ann in enumerate(train_anns[:3]):
    print(f"  {i+1}. {ann['description'][:80]}")

READ16_LOADED = True

# Guard: READ16 competition test set sometimes has no transcriptions.
# If test_anns is empty, fall back to validation set for final evaluation.
if len(test_anns) == 0:
    print("\n⚠️  WARNING: Test split has 0 annotated lines.")
    print("   This is normal for the ICFHR-2016 competition release — test labels are held out.")
    print("   Falling back to VALIDATION set for final evaluation.")
    test_anns   = val_anns
    test_images = val_images
    USING_VAL_AS_TEST = True
else:
    USING_VAL_AS_TEST = False
    print(f"\n✅ Test split has {len(test_anns):,} annotated lines.")

print("\n✅ READ16 dataset ready!")


# In[6]:


# ===================================================================
# CELL 6: Data Augmentation
# ===================================================================

class ManuscriptAugmentation:
    def __call__(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image)

        # Random slight rotation
        if random.random() < 0.4:
            angle = random.uniform(-5, 5)
            h, w = img_np.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img_np = cv2.warpAffine(img_np, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # Random scale
        if random.random() < 0.3:
            scale = random.uniform(0.95, 1.05)
            h, w = img_np.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            img_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            if scale > 1:
                sy, sx = (new_h - h) // 2, (new_w - w) // 2
                img_np = img_np[sy:sy+h, sx:sx+w]
            else:
                py, px = (h - new_h) // 2, (w - new_w) // 2
                img_np = cv2.copyMakeBorder(
                    img_np, py, h-new_h-py, px, w-new_w-px, cv2.BORDER_REPLICATE)

        # Brightness
        if random.random() < 0.3:
            factor = random.uniform(0.8, 1.2)
            img_np = np.clip(img_np.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        # Contrast
        if random.random() < 0.3:
            factor = random.uniform(0.9, 1.1)
            mean = img_np.mean()
            img_np = np.clip((img_np - mean) * factor + mean, 0, 255).astype(np.uint8)

        return Image.fromarray(img_np)

print("✅ ManuscriptAugmentation defined!")


# In[7]:


# ===================================================================
# CELL 7: TrOCR Dataset, Metrics, and Data Collator
# ===================================================================
from transformers import TrOCRProcessor
import editdistance
from jiwer import wer as jiwer_wer

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")


class TrOCRDataset(torch.utils.data.Dataset):
    """
    Loads cropped text-line images from page images using COCO-style annotations.
    Each annotation contains a bounding box (bbox) and a transcription (description).
    `image_dir_map` maps image_id → Path of the directory containing that image.
    """

    def __init__(self, annotations, image_map, image_dir_map, processor,
                 max_target_length=128, augment_transform=None, use_clahe=False):
        self.annotations      = annotations
        self.image_map        = image_map
        self.image_dir_map    = image_dir_map   # NEW: per-image-id directory lookup
        self.processor        = processor
        self.max_target_length = max_target_length
        self.augment_transform = augment_transform
        self.use_clahe        = use_clahe
        self._clahe           = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self):
        return len(self.annotations)

    def _apply_clahe(self, img_np):
        if len(img_np.shape) == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        enhanced = self._clahe.apply(img_np)
        return Image.fromarray(enhanced).convert("RGB")

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_info = self.image_map[ann["image_id"]]
        img_dir  = self.image_dir_map[ann["image_id"]]
        img_path = img_dir / img_info["file_name"]

        image = Image.open(img_path).convert("RGB")
        x, y, w, h = ann["bbox"]
        # Guard against zero-size crops
        x, y = max(0, x), max(0, y)
        w, h = max(1, w), max(1, h)
        crop = image.crop((x, y, x + w, y + h))

        if self.use_clahe:
            crop = self._apply_clahe(np.array(crop))

        if self.augment_transform is not None:
            crop = self.augment_transform(crop)

        encoding = self.processor(crop, return_tensors="pt",
                                  padding="max_length", truncation=True)
        pixel_values = encoding["pixel_values"].squeeze(0)

        labels = self.processor.tokenizer(
            ann.get("description", ""),
            padding="max_length", truncation=True,
            max_length=self.max_target_length, return_tensors="pt"
        )["input_ids"].squeeze(0)

        return {"pixel_values": pixel_values, "labels": labels}


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids   = pred.predictions

    pred_str  = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = float(np.mean([
        editdistance.eval(p, l) / max(len(l), 1)
        for p, l in zip(pred_str, label_str)
    ]))
    wer = jiwer_wer(label_str, pred_str)
    return {"cer": cer, "wer": wer}


class FixedDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels       = torch.stack([item["labels"] for item in batch])
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


print("✅ Dataset, metrics, and collator defined!")


# In[8]:


# ===================================================================
# CELL 8: Layer Freezing Configuration
# ===================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_freezing(model, freeze_encoder, freeze_decoder, verbose=True):
    num_encoder_layers = len(model.encoder.encoder.layer)
    num_decoder_layers = len(model.decoder.model.decoder.layers)

    print(f"\n🔍 Detected layers: encoder={num_encoder_layers}, decoder={num_decoder_layers}")

    for i, layer in enumerate(model.encoder.encoder.layer):
        for p in layer.parameters():
            p.requires_grad = (i >= freeze_encoder)

    for i, layer in enumerate(model.decoder.model.decoder.layers):
        for p in layer.parameters():
            p.requires_grad = (i >= freeze_decoder)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())

    config = {
        "freeze_encoder": freeze_encoder,
        "freeze_decoder": freeze_decoder,
        "total_encoder_layers": num_encoder_layers,
        "total_decoder_layers": num_decoder_layers,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio": trainable / total,
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"Freezing Configuration:")
        print(f"  Encoder: {freeze_encoder}/{num_encoder_layers} layers frozen")
        print(f"  Decoder: {freeze_decoder}/{num_decoder_layers} layers frozen")
        print(f"  Trainable params: {trainable:,}/{total:,} ({100*trainable/total:.1f}%)")
        print(f"{'='*70}\n")

    return config

print("✅ Layer freezing functions defined!")


# In[9]:


# ===================================================================
# CELL 9: One-Cycle Trainer & Main Training Function
# ===================================================================
from transformers import VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import EarlyStoppingCallback
import torch


class _HuttnerOneCycleTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer with AdamW + OneCycleLR (paper-style scheduling)."""

    def __init__(self, *args,
                 onecycle_max_lr, onecycle_pct_start,
                 onecycle_base_momentum, onecycle_max_momentum,
                 onecycle_initial_lr, onecycle_final_div_factor,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._onecycle_max_lr          = float(onecycle_max_lr)
        self._onecycle_pct_start       = float(onecycle_pct_start)
        self._onecycle_base_momentum   = float(onecycle_base_momentum)
        self._onecycle_max_momentum    = float(onecycle_max_momentum)
        self._onecycle_initial_lr      = float(onecycle_initial_lr)
        self._onecycle_final_div_factor = float(onecycle_final_div_factor)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params, lr=self._onecycle_max_lr, weight_decay=1e-4)
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is not None:
            return self.lr_scheduler
        optimizer = optimizer or self.optimizer
        div_factor = self._onecycle_max_lr / self._onecycle_initial_lr
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self._onecycle_max_lr,
            total_steps=num_training_steps,
            pct_start=self._onecycle_pct_start,
            anneal_strategy="cos",
            cycle_momentum=True,
            base_momentum=self._onecycle_base_momentum,
            max_momentum=self._onecycle_max_momentum,
            div_factor=div_factor,
            final_div_factor=self._onecycle_final_div_factor,
        )
        return self.lr_scheduler




def run_single_experiment(exp_config, seed=SEED, use_clahe=USE_CLAHE, use_aug=USE_AUG):
    set_seed(seed)

    exp_name  = exp_config['name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = OUTPUT_BASE / f"{exp_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# Starting  : {exp_name}")
    print(f"# Dataset   : READ16  (official train/val/test splits)")
    print(f"# Encoder   : {exp_config['freeze_encoder']}/12 layers frozen")
    print(f"# Decoder   : {exp_config['freeze_decoder']}/6  layers frozen")
    print(f"# One-Cycle : {USE_ONECYCLE}   CLAHE: {use_clahe}   Aug: {use_aug}")
    print(f"{'#'*70}\n")

    # Load model
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    model.to(device)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id

    # Freeze layers
    config = configure_freezing(
        model,
        freeze_encoder=exp_config['freeze_encoder'],
        freeze_decoder=exp_config['freeze_decoder'],
        verbose=True,
    )

    config.update({
        "experiment_name"  : exp_config['name'],
        "dataset"          : "READ16",
        "seed"             : seed,
        "use_clahe"        : use_clahe,
        "use_aug"          : use_aug,
        "num_epochs"       : NUM_EPOCHS,
        "learning_rate"    : LR,
        "use_onecycle"     : USE_ONECYCLE,
        "onecycle_max_lr"  : ONECYCLE_MAX_LR  if USE_ONECYCLE else None,
        "onecycle_pct_start": ONECYCLE_PCT_START if USE_ONECYCLE else None,
    })

    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Build datasets (use image_dir_map so each split reads from its own folder)
    aug = ManuscriptAugmentation() if use_aug else None

    train_dataset = TrOCRDataset(
        train_anns, image_map, image_dir_map, processor,
        max_target_length=128, augment_transform=aug, use_clahe=use_clahe)

    val_dataset = TrOCRDataset(
        val_anns, image_map, image_dir_map, processor,
        max_target_length=128, augment_transform=None, use_clahe=use_clahe)

    test_dataset = TrOCRDataset(
        test_anns, image_map, image_dir_map, processor,
        max_target_length=128, augment_transform=None, use_clahe=use_clahe)

    print(f"Dataset sizes — Train: {len(train_dataset)}, "
          f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Training arguments
    common_args = dict(
        output_dir=str(out_dir / "checkpoints"),
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=2,
        dataloader_num_workers=4,   # Colab typically has 2 CPUs
        report_to="none",
        seed=seed,
        # Speed-up flags
        dataloader_pin_memory=True,
    )

    if USE_ONECYCLE:
        training_args = Seq2SeqTrainingArguments(
            **common_args,
            learning_rate=ONECYCLE_MAX_LR,
            warmup_ratio=0.0,
            weight_decay=0.0,
            label_smoothing_factor=0.0,
            fp16=False,   # Keep False with OneCycle: very small initial LR (1e-9) can cause
                          # AMP gradient scaler overflow before the LR ramps up
        )
    else:
        training_args = Seq2SeqTrainingArguments(
            **common_args,
            learning_rate=LR,
            warmup_ratio=0.1,
            weight_decay=0.01,
            label_smoothing_factor=0.1,
            fp16=True,    # Safe to use fp16 with standard warmup LR
        )

    # Build trainer
    if USE_ONECYCLE:
        model.generation_config.max_length          = 128
        model.generation_config.num_beams           = 4
        model.generation_config.early_stopping      = True
        model.generation_config.no_repeat_ngram_size = 0
        model.generation_config.length_penalty      = 1.0

        trainer = _HuttnerOneCycleTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=FixedDataCollator(processor),
            compute_metrics=compute_metrics,
            callbacks=[],
            onecycle_max_lr=ONECYCLE_MAX_LR,
            onecycle_pct_start=ONECYCLE_PCT_START,
            onecycle_base_momentum=ONECYCLE_BASE_MOMENTUM,
            onecycle_max_momentum=ONECYCLE_MAX_MOMENTUM,
            onecycle_initial_lr=ONECYCLE_INITIAL_LR,
            onecycle_final_div_factor=ONECYCLE_FINAL_DIV_FACTOR,
        )
    else:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=FixedDataCollator(processor),
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
        )

    # --- Train ---
    print("\n🚀 Starting training...")
    start_time = time.time()
    train_result = trainer.train()
    training_time = time.time() - start_time

    # --- Evaluate on test set ---
    print("\n📊 Evaluating on test set...")
    test_result = trainer.predict(test_dataset)

    # Compile & save results
    results = {
        "experiment"         : exp_name,
        "dataset"            : "READ16",
        "freeze_encoder"     : exp_config['freeze_encoder'],
        "freeze_decoder"     : exp_config['freeze_decoder'],
        "seed"               : seed,
        "config"             : config,
        "train_metrics"      : {k: float(v) if isinstance(v, (int, float, np.number)) else v
                                for k, v in train_result.metrics.items()},
        "test_metrics"       : {k: float(v) if isinstance(v, (int, float, np.number)) else v
                                for k, v in test_result.metrics.items()},
        "training_time_seconds": training_time,
        "output_dir"         : str(out_dir),
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    model_dir = out_dir / "final_model"
    trainer.save_model(model_dir)
    processor.save_pretrained(model_dir)

    print(f"\n✅ Results for {exp_name}:")
    print(f"   Test CER       : {test_result.metrics['test_cer']:.4f}")
    print(f"   Test WER       : {test_result.metrics['test_wer']:.4f}")
    print(f"   Training time  : {training_time/60:.1f} min")
    print(f"   Saved to       : {out_dir}")

    del model, trainer, train_dataset, val_dataset, test_dataset
    torch.cuda.empty_cache()
    return results

print("✅ One-Cycle Trainer and training function defined!")


# In[10]:


get_ipython().run_cell_magic('capture', 'stored_output1', '\n# ===================================================================\n# CELL 10: RUN SELECTED CONFIGURATION\n# ===================================================================\nprint("\\n" + "="*80)\nprint("🎯 RUNNING SELECTED CONFIGURATION")\nprint("="*80)\n\nresult = run_single_experiment(\n    exp_config=SELECTED_CONFIG,\n    seed=SEED,\n    use_clahe=USE_CLAHE,\n    use_aug=USE_AUG,\n)\n\nprint(f"\\n✅ Training complete for {SELECTED_CONFIG[\'name\']}")\nprint(f"   Test CER: {result[\'test_metrics\'][\'test_cer\']:.4f}")\nprint(f"   Test WER: {result[\'test_metrics\'][\'test_wer\']:.4f}")\n')


# In[11]:


stored_output1.show()


# In[12]:


get_ipython().run_cell_magic('capture', 'stored_output2', '\n# ===================================================================\n# CELL 11: COMPUTE LINE-WISE CER FOR EACH TEST SAMPLE\n# ===================================================================\nfrom tqdm import tqdm\nfrom transformers import VisionEncoderDecoderModel, TrOCRProcessor\n\nprint("\\n" + "="*80)\nprint("📊 COMPUTING LINE-WISE CER FOR EACH TEST SAMPLE")\nprint("="*80)\n\nout_dir    = Path(result[\'output_dir\'])\nmodel_dir  = out_dir / "final_model"\nprint(f"\\n   Loading model from: {model_dir}")\n\nmodel_eval        = VisionEncoderDecoderModel.from_pretrained(model_dir)\nprocessor_reload  = TrOCRProcessor.from_pretrained(model_dir)\nmodel_eval.to(device)\nmodel_eval.eval()\n\ntest_dataset_eval = TrOCRDataset(\n    test_anns, image_map, image_dir_map, processor_reload,\n    max_target_length=128, augment_transform=None, use_clahe=USE_CLAHE\n)\nprint(f"   Test dataset size: {len(test_dataset_eval)} samples")\n\nline_wise_results = []\nprint("\\n🔍 Generating predictions for each test sample...")\n\nfor idx in tqdm(range(len(test_dataset_eval)), desc="Processing"):\n    sample = test_dataset_eval[idx]\n    ann    = test_anns[idx]\n    ground_truth = ann.get("description", "")\n\n    pixel_values = sample["pixel_values"].unsqueeze(0).to(device)\n    with torch.no_grad():\n        generated_ids = model_eval.generate(pixel_values)\n    prediction = processor_reload.batch_decode(generated_ids, skip_special_tokens=True)[0]\n\n    if len(ground_truth) == 0:\n        line_cer = 0.0 if len(prediction) == 0 else 1.0\n        edit_dist = 0\n    else:\n        edit_dist = editdistance.eval(prediction, ground_truth)\n        line_cer  = edit_dist / len(ground_truth)\n\n    line_wise_results.append({\n        "sample_idx"   : idx,\n        "image_id"     : ann["image_id"],\n        "bbox"         : ann["bbox"],\n        "ground_truth" : ground_truth,\n        "prediction"   : prediction,\n        "cer"          : line_cer,\n        "edit_distance": edit_dist,\n        "gt_length"    : len(ground_truth),\n        "pred_length"  : len(prediction),\n    })\n\n# Statistics\nif len(line_wise_results) == 0:\n    raise RuntimeError("No predictions generated. Check that test_anns is not empty.")\n\ntotal_cer     = sum(r["cer"] for r in line_wise_results) / len(line_wise_results)\ntotal_edit    = sum(r["edit_distance"] for r in line_wise_results)\ntotal_gt_ch   = sum(r["gt_length"] for r in line_wise_results)\noverall_cer   = total_edit / total_gt_ch if total_gt_ch > 0 else 0\ncer_vals      = sorted(r["cer"] for r in line_wise_results)\n\nprint(f"\\n📈 LINE-WISE CER STATISTICS:")\nprint("=" * 80)\nprint(f"Total test samples        : {len(line_wise_results)}")\nprint(f"Average per-line CER      : {total_cer:.4f}")\nprint(f"Overall CER (edit/chars)  : {overall_cer:.4f}")\nprint(f"Min CER                   : {min(cer_vals):.4f}")\nprint(f"Max CER                   : {max(cer_vals):.4f}")\nprint(f"Median CER                : {cer_vals[len(cer_vals)//2]:.4f}")\nprint("=" * 80)\n\n# Save JSON\noutput_data = {\n    "configuration": SELECTED_CONFIG,\n    "dataset": "READ16",\n    "statistics": {\n        "total_samples"      : len(line_wise_results),\n        "average_per_line_cer": total_cer,\n        "overall_cer"        : overall_cer,\n        "min_cer"            : min(cer_vals),\n        "max_cer"            : max(cer_vals),\n        "median_cer"         : cer_vals[len(cer_vals)//2],\n        "total_edit_distance": total_edit,\n        "total_gt_chars"     : total_gt_ch,\n    },\n    "per_line_results": line_wise_results,\n}\n\nlinewise_file = out_dir / "linewise_cer_results.json"\nwith open(linewise_file, "w", encoding="utf-8") as f:\n    json.dump(output_data, f, indent=2, ensure_ascii=False)\n\n# Save CSV\ndf = pd.DataFrame(line_wise_results)\ncsv_file = out_dir / "linewise_cer_results.csv"\ndf.to_csv(csv_file, index=False, encoding="utf-8")\n\nprint(f"\\n✅ Saved:")\nprint(f"   {linewise_file}")\nprint(f"   {csv_file}")\n\n# Sample predictions\nprint(f"\\n📝 SAMPLE PREDICTIONS (first 10):")\nprint("=" * 80)\nfor i, res in enumerate(line_wise_results[:10], 1):\n    print(f"\\nSample {i}:")\n    print(f"  GT  : {res[\'ground_truth\']}")\n    print(f"  Pred: {res[\'prediction\']}")\n    print(f"  CER : {res[\'cer\']:.4f}")\n\nprint("\\n" + "="*80)\nprint("🎉 COMPLETE!")\nprint("="*80)\nprint(f"\\n✅ All files saved to: {out_dir}")\n\ndel model_eval\ntorch.cuda.empty_cache()\n')


# In[13]:


stored_output2.show()

