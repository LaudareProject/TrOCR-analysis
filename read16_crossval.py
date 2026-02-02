# ===================================================================
# CELL 1: Setup and Dependencies
# ===================================================================

import os
import json
import random
import tarfile
import zipfile
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset

print("✅ Dependencies imported!")

# ===================================================================
# CELL 2: Extract READ16.zip
# ===================================================================

# Get the script directory
SCRIPT_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# Define paths relative to script location
ZIP_FILE = SCRIPT_DIR / "READ16.zip"
EXTRACT_DIR = SCRIPT_DIR / "READ16-data"

# Check if data is already extracted (check for a specific file to be sure)
train_check = EXTRACT_DIR / "PublicData" / "Training" / "Images"
if train_check.exists() and len(list(train_check.glob("*.JPG"))) > 0:
    print(f"✅ Dataset already extracted at {EXTRACT_DIR}")
else:
    # Check if the .zip file exists
    if not ZIP_FILE.exists():
        raise FileNotFoundError(
            f"❌ READ16.zip not found at {ZIP_FILE}\n"
            f"Please download the dataset and place it in {SCRIPT_DIR}"
        )

    # Create extraction directory
    EXTRACT_DIR.mkdir(exist_ok=True)
    temp_extract = SCRIPT_DIR / "READ16-data-temp"
    temp_extract.mkdir(exist_ok=True)

    try:
        # Step 1: Extract ZIP file
        print(f"📦 Extracting {ZIP_FILE.name}...")
        with zipfile.ZipFile(ZIP_FILE, "r") as z:
            z.extractall(temp_extract)
        print(f"✅ ZIP extracted!")

        # Step 2: Extract the tar.gz files from the ZIP
        tgz_files = list(temp_extract.glob("*.tgz"))
        print(f"\n📦 Found {len(tgz_files)} tar.gz files to extract...")

        for tgz_file in tgz_files:
            print(f"   Extracting {tgz_file.name}...")
            with tarfile.open(tgz_file, "r:gz") as tar:
                tar.extractall(EXTRACT_DIR)
            print(f"   ✅ {tgz_file.name} extracted!")

        # Cleanup temp directory
        shutil.rmtree(temp_extract)
        print(f"\n✅ Extraction complete!")

    except Exception as e:
        # Cleanup on error
        if temp_extract.exists():
            shutil.rmtree(temp_extract)
        raise RuntimeError(f"❌ Failed to extract archive: {e}")

# List extracted contents
print("\n📁 Dataset structure:")
for item in sorted(EXTRACT_DIR.iterdir()):
    if item.is_dir():
        print(f"   📁 {item.name}/")

# ===================================================================
# CELL 3: Configuration and Path Setup
# ===================================================================

# Configuration
# Use the extracted READ16-data directory
LOCAL_READ_BOZEN = EXTRACT_DIR
OUTPUT_BASE = SCRIPT_DIR / "bozen_training_output"

OUTPUT_BASE.mkdir(exist_ok=True)

# Set up dataset paths based on extracted structure
BOZEN_TRAIN_IMG_DIR = LOCAL_READ_BOZEN / "PublicData" / "Training" / "Images"
BOZEN_TRAIN_PAGE_DIR = LOCAL_READ_BOZEN / "PublicData" / "Training" / "page"

BOZEN_VAL_IMG_DIR = LOCAL_READ_BOZEN / "PublicData" / "Validation" / "Images"
BOZEN_VAL_PAGE_DIR = LOCAL_READ_BOZEN / "PublicData" / "Validation" / "page"

BOZEN_TEST_IMG_DIR = LOCAL_READ_BOZEN / "Test-ICFHR-2016"
BOZEN_TEST_PAGE_DIR = LOCAL_READ_BOZEN / "Test-ICFHR-2016" / "page"

# Count images in each split
print("\n📊 Dataset Statistics:")

train_img_count = len(list(BOZEN_TRAIN_IMG_DIR.glob("*.JPG"))) + len(
    list(BOZEN_TRAIN_IMG_DIR.glob("*.jpg"))
)
train_xml_count = len(list(BOZEN_TRAIN_PAGE_DIR.glob("**/*.xml")))
print(f"   TRAIN: {train_img_count} images, {train_xml_count} XML files")
print(f"          Images: {BOZEN_TRAIN_IMG_DIR}")
print(f"          XMLs:   {BOZEN_TRAIN_PAGE_DIR}")

val_img_count = len(list(BOZEN_VAL_IMG_DIR.glob("*.JPG"))) + len(
    list(BOZEN_VAL_IMG_DIR.glob("*.jpg"))
)
val_xml_count = len(list(BOZEN_VAL_PAGE_DIR.glob("**/*.xml")))
print(f"   VAL:   {val_img_count} images, {val_xml_count} XML files")
print(f"          Images: {BOZEN_VAL_IMG_DIR}")
print(f"          XMLs:   {BOZEN_VAL_PAGE_DIR}")

test_img_count = len(list(BOZEN_TEST_IMG_DIR.glob("*.JPG"))) + len(
    list(BOZEN_TEST_IMG_DIR.glob("*.jpg"))
)
test_xml_count = len(list(BOZEN_TEST_PAGE_DIR.glob("**/*.xml")))
print(f"   TEST:  {test_img_count} images, {test_xml_count} XML files")
print(f"          Images: {BOZEN_TEST_IMG_DIR}")
print(f"          XMLs:   {BOZEN_TEST_PAGE_DIR}")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# Enable CUDA error checking
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Clear CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("✅ Configuration complete!")

# ===================================================================
# CELL 4: PAGE XML Parser
# ===================================================================


def parse_page_xml(xml_file):
    """
    Extract text lines from PAGE XML format
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # PAGE XML namespaces
        namespaces = {
            "page": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15",
            "page2010": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19",
        }

        # Find Page element
        page = None
        ns = None
        for ns_name, ns_uri in namespaces.items():
            page = root.find(f".//{{{ns_uri}}}Page")
            if page is not None:
                ns = {ns_name: ns_uri}
                ns_prefix = ns_name
                break

        if page is None:
            page = root.find(".//Page")
            ns = {}
            ns_prefix = None

        if page is None:
            return None, []

        # Get image filename
        image_filename = page.get("imageFilename")

        # Extract text lines
        text_lines = []

        if ns_prefix:
            text_line_elements = root.findall(f".//{{{ns[ns_prefix]}}}TextLine")
        else:
            text_line_elements = root.findall(".//TextLine")

        for text_line in text_line_elements:
            if ns_prefix:
                coords_elem = text_line.find(f".//{{{ns[ns_prefix]}}}Coords")
                unicode_elem = text_line.find(f".//{{{ns[ns_prefix]}}}Unicode")
            else:
                coords_elem = text_line.find(".//Coords")
                unicode_elem = text_line.find(".//Unicode")

            coords_str = coords_elem.get("points") if coords_elem is not None else None
            text = (
                unicode_elem.text
                if unicode_elem is not None and unicode_elem.text
                else ""
            )

            if text.strip() and coords_str:
                try:
                    points = coords_str.strip().split()
                    coords = [list(map(int, map(float, p.split(",")))) for p in points]

                    xs = [c[0] for c in coords]
                    ys = [c[1] for c in coords]
                    x, y = min(xs), min(ys)
                    w, h = max(xs) - x, max(ys) - y

                    text_lines.append({"text": text.strip(), "bbox": [x, y, w, h]})
                except:
                    continue

        return image_filename, text_lines

    except Exception as e:
        return None, []


def load_read_bozen_dataset(page_dir, img_dir):
    """
    Load READ Bozen dataset in COCO format
    """
    print(f"\n📖 Loading READ Bozen dataset...")

    images = []
    annotations = []
    ann_id = 0

    page_files = sorted(page_dir.glob("**/*.xml"))
    print(f"   Found {len(page_files)} PAGE XML files")

    skipped_no_img = 0
    skipped_no_text = 0

    for img_id, page_file in enumerate(page_files):
        if img_id % 50 == 0 and img_id > 0:
            print(f"   Processing: {img_id}/{len(page_files)} pages...")

        image_filename, text_lines = parse_page_xml(page_file)

        if not image_filename or not text_lines:
            skipped_no_text += 1
            continue

        # Find image file
        img_path = img_dir / image_filename
        if not img_path.exists():
            base_name = Path(image_filename).stem
            for ext in [".jpg", ".JPG", ".png", ".PNG"]:
                test_path = img_dir / f"{base_name}{ext}"
                if test_path.exists():
                    img_path = test_path
                    image_filename = img_path.name
                    break

        if not img_path.exists():
            skipped_no_img += 1
            continue

        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            skipped_no_img += 1
            continue

        h, w = img.shape[:2]

        images.append(
            {"id": img_id, "file_name": image_filename, "width": w, "height": h}
        )

        for line_data in text_lines:
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 6,
                    "bbox": line_data["bbox"],
                    "description": line_data["text"],
                }
            )
            ann_id += 1

    print(f"\n✅ Loaded: {len(images)} pages, {len(annotations)} text lines")
    print(f"⚠️  Skipped: {skipped_no_text} (no text), {skipped_no_img} (no image)")

    return images, annotations


print("✅ XML parser functions defined!")

# ===================================================================
# CELL 5: Load READ Bozen Dataset
# ===================================================================

if not (
    BOZEN_TRAIN_IMG_DIR.exists()
    and BOZEN_VAL_IMG_DIR.exists()
    and BOZEN_TEST_IMG_DIR.exists()
):
    raise RuntimeError("Dataset not ready. Check paths in Cell 3.")

# Load each split separately
print("\n📖 Loading TRAIN split...")
bozen_train_images, bozen_train_annotations = load_read_bozen_dataset(
    BOZEN_TRAIN_PAGE_DIR, BOZEN_TRAIN_IMG_DIR
)

print("\n📖 Loading VALIDATION split...")
bozen_val_images, bozen_val_annotations = load_read_bozen_dataset(
    BOZEN_VAL_PAGE_DIR, BOZEN_VAL_IMG_DIR
)

print("\n📖 Loading TEST split...")
bozen_test_images, bozen_test_annotations = load_read_bozen_dataset(
    BOZEN_TEST_PAGE_DIR, BOZEN_TEST_IMG_DIR
)

# Create image maps for each split
bozen_train_image_map = {img["id"]: img for img in bozen_train_images}
bozen_val_image_map = {img["id"]: img for img in bozen_val_images}
bozen_test_image_map = {img["id"]: img for img in bozen_test_images}

print(f"\n{'=' * 70}")
print("PRE-SPLIT DATASET LOADED")
print(f"{'=' * 70}")
print(
    f"Train: {len(bozen_train_annotations)} text lines from {len(bozen_train_images)} pages"
)
print(
    f"Val:   {len(bozen_val_annotations)} text lines from {len(bozen_val_images)} pages"
)
print(
    f"Test:  {len(bozen_test_annotations)} text lines from {len(bozen_test_images)} pages"
)
print(f"{'=' * 70}\n")

# Show samples from each split
print("Sample text lines from TRAIN:")
for i in range(min(3, len(bozen_train_annotations))):
    sample = bozen_train_annotations[i]
    print(f"  {i + 1}. {sample['description'][:60]}...")

BOZEN_LOADED = True
print("\n✅ READ Bozen pre-split dataset ready!")

# ===================================================================
# CELL 6: Data Augmentation (Same as Cortonese)
# ===================================================================


class ManuscriptAugmentation:
    def __call__(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image)

        # Rotation
        if random.random() < 0.4:
            angle = random.uniform(-5, 5)
            h, w = img_np.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img_np = cv2.warpAffine(img_np, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # Elastic deformation
        if random.random() < 0.25:
            h, w = img_np.shape[:2]
            if h > 20 and w > 20:
                dx = (
                    cv2.resize(
                        np.random.randn(max(h // 10, 3), max(w // 10, 3)), (w, h)
                    )
                    * 3
                )
                dy = (
                    cv2.resize(
                        np.random.randn(max(h // 10, 3), max(w // 10, 3)), (w, h)
                    )
                    * 3
                )
                x, y = np.meshgrid(np.arange(w), np.arange(h))
                map_x = np.clip(x + dx, 0, w - 1).astype(np.float32)
                map_y = np.clip(y + dy, 0, h - 1).astype(np.float32)
                img_np = cv2.remap(img_np, map_x, map_y, cv2.INTER_LINEAR)

        # Gaussian blur
        if random.random() < 0.35:
            kernel_size = random.choice([3, 5, 7])
            sigma = random.uniform(0.5, 1.2)
            img_np = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), sigma)

        # Brightness
        if random.random() < 0.4:
            factor = random.uniform(0.7, 1.3)
            img_np = np.clip(img_np * factor, 0, 255).astype(np.uint8)

        # Contrast
        if random.random() < 0.3:
            factor = random.uniform(0.8, 1.2)
            mean = img_np.mean()
            img_np = np.clip((img_np - mean) * factor + mean, 0, 255).astype(np.uint8)

        # Speckle noise
        if random.random() < 0.25:
            noise = np.random.randn(*img_np.shape) * 6
            img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)

        # Morphological ops
        if random.random() < 0.15:
            kernel = np.ones((2, 2), np.uint8)
            if random.random() < 0.5:
                img_np = cv2.erode(img_np, kernel, iterations=1)
            else:
                img_np = cv2.dilate(img_np, kernel, iterations=1)

        # Shadow/staining
        if random.random() < 0.15:
            h, w = img_np.shape[:2]
            if h > 10 and w > 10:
                shadow = np.random.randn(max(h // 5, 2), max(w // 5, 2))
                shadow = cv2.resize(shadow, (w, h))
                shadow = (shadow - shadow.min()) / (shadow.max() - shadow.min() + 1e-8)
                shadow = 1 - shadow * 0.25
                if len(img_np.shape) == 3:
                    shadow = np.expand_dims(shadow, axis=-1)
                img_np = np.clip(img_np * shadow, 0, 255).astype(np.uint8)

        return Image.fromarray(img_np)


print("✅ Augmentation class defined!")

# ===================================================================
# CELL 7: TrOCR Dataset Class
# ===================================================================

from transformers import TrOCRProcessor


class TrOCRDataset(Dataset):
    def __init__(
        self,
        annotations,
        images_info,
        image_root,
        processor: TrOCRProcessor,
        max_target_length=128,
        augment_transform=None,
        use_clahe=True,
    ):
        self.annotations = annotations
        self.image_root = Path(image_root)
        self.processor = processor
        self.max_target_length = max_target_length
        self.augment_transform = augment_transform
        self.use_clahe = use_clahe
        self.image_id_to_info = {img["id"]: img for img in images_info}

    def preprocess_image(self, img_np):
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_np

        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
        else:
            enhanced = cv2.equalizeHist(gray)

        rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        return rgb

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        text = ann.get("description", "") or " "
        img_info = self.image_id_to_info[ann["image_id"]]
        image_path = self.image_root / img_info["file_name"]

        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None

            x, y, w, h = [int(v) for v in ann["bbox"]]
            margin = 8
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(img.shape[1] - x, w + 2 * margin)
            h = min(img.shape[0] - y, h + 2 * margin)

            crop = img[y : y + h, x : x + w]
            crop = self.preprocess_image(crop)
            pil = Image.fromarray(crop)

            if self.augment_transform:
                pil = self.augment_transform(pil)

            # Process image
            pixel_values = self.processor(
                images=pil, return_tensors="pt"
            ).pixel_values.squeeze(0)

            # Process text - CRITICAL FIX: squeeze to get [128] shape
            labels = self.processor.tokenizer(
                text,
                padding="max_length",
                max_length=self.max_target_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.squeeze(0)

            return {"pixel_values": pixel_values, "labels": labels}

        except Exception as e:
            # Log the error and return None
            print(f"Error processing item {idx}: {e}")
            return None


print("✅ Dataset class defined!")

# ===================================================================
# CELL 8: Data Collator
# ===================================================================


class FixedDataCollator:
    """Data collator that properly creates decoder_input_ids for TrOCR training"""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        # Filter None values
        features = [f for f in features if f is not None]

        if not features:
            return {}

        # Stack pixel values
        pixel_values = torch.stack([f["pixel_values"] for f in features])

        # Stack labels
        labels = torch.stack([f["labels"] for f in features])

        # Create decoder_input_ids by shifting labels right
        # This is what the model expects for teacher forcing during training
        decoder_input_ids = labels.clone()

        # Shift right: move everything one position to the right
        decoder_input_ids[:, 1:] = labels[:, :-1].clone()

        # Set the first token to decoder_start_token_id (usually BOS token)
        decoder_input_ids[:, 0] = self.processor.tokenizer.cls_token_id

        # Replace pad tokens in decoder_input_ids with pad_token_id (not -100)
        decoder_input_ids[decoder_input_ids == -100] = (
            self.processor.tokenizer.pad_token_id
        )

        # Keep labels as-is (with -100 for padding) for loss calculation
        labels_for_loss = labels.clone()
        labels_for_loss[labels_for_loss == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels_for_loss,
        }


print("✅ Data collator defined!")

# ===================================================================
# CELL 9: Evaluation Metrics
# ===================================================================

import editdistance
from jiwer import wer as jiwer_wer


def calculate_wer_cer(pred_strs, label_strs):
    # Filter out empty strings from both lists simultaneously
    filtered_pairs = [
        (p, l) for p, l in zip(pred_strs, label_strs) if p.strip() or l.strip()
    ]
    if not filtered_pairs:
        return 1.0, 1.0

    pred_strs_filtered, label_strs_filtered = zip(*filtered_pairs)

    total_cer = 0.0
    for p, l in zip(pred_strs_filtered, label_strs_filtered):
        total_cer += editdistance.eval(p, l) / max(len(l), 1)
    cer = total_cer / len(pred_strs_filtered)

    try:
        wer = jiwer_wer(list(label_strs_filtered), list(pred_strs_filtered))
    except:
        wer = 1.0

    return wer, cer


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_ids[pred_ids < 0] = processor.tokenizer.pad_token_id
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id

    pred_strs = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_strs = processor.batch_decode(labels_ids, skip_special_tokens=True)

    wer, cer = calculate_wer_cer(pred_strs, label_strs)
    return {"wer": wer, "cer": cer}


print("✅ Metrics defined!")

# ===================================================================
# CELL 10: Train TrOCR on READ Bozen
# ===================================================================

from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)

print("\n" + "#" * 70)
print("# PUBLIC DATASET VALIDATION: READ Bozen (Pre-split)")
print("# Configuration: Full FT + CLAHE + Augmentation")
print("#" * 70 + "\n")

# Load model
MODEL_NAME = "microsoft/trocr-base-handwritten"
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

# Configure
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = len(processor.tokenizer)
model.config.max_length = 128

# Full fine-tuning
for param in model.parameters():
    param.requires_grad = True

model.to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable:,} (100%)\n")

# Create datasets - NOTE: Each split uses its own image directory
aug = ManuscriptAugmentation()

train_dataset = TrOCRDataset(
    bozen_train_annotations,
    bozen_train_images,
    BOZEN_TRAIN_IMG_DIR,  # Train images
    processor,
    max_target_length=128,
    augment_transform=aug,
    use_clahe=True,
)

val_dataset = TrOCRDataset(
    bozen_val_annotations,
    bozen_val_images,
    BOZEN_VAL_IMG_DIR,  # Validation images
    processor,
    max_target_length=128,
    augment_transform=None,
    use_clahe=True,
)

test_dataset = TrOCRDataset(
    bozen_test_annotations,
    bozen_test_images,
    BOZEN_TEST_IMG_DIR,  # Test images
    processor,
    max_target_length=128,
    augment_transform=None,
    use_clahe=True,
)

print(
    f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}\n"
)

# Training args
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = OUTPUT_BASE / f"bozen_presplit_{timestamp}"
out_dir.mkdir(parents=True, exist_ok=True)

training_args = Seq2SeqTrainingArguments(
    output_dir=str(out_dir / "checkpoints"),
    predict_with_generate=True,
    generation_max_length=128,
    generation_num_beams=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=50,
    learning_rate=3e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    label_smoothing_factor=0.1,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    logging_strategy="steps",
    logging_steps=50,
    save_total_limit=3,
    dataloader_num_workers=2,
    report_to="none",
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=FixedDataCollator(processor),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

# Train
print("🚀 Starting training...\n")
try:
    train_result = trainer.train()

    # Evaluate on official test set
    print("\n📊 Evaluating on official test set...\n")
    test_result = trainer.predict(test_dataset)

    # Save
    results = {
        "dataset": "READ Bozen (1470-1805) - Pre-split",
        "splits": {
            "train": f"{len(train_dataset)} lines",
            "val": f"{len(val_dataset)} lines",
            "test": f"{len(test_dataset)} lines",
        },
        "config": {"use_clahe": True, "use_aug": True, "freeze_layers": 0},
        "train_metrics": train_result.metrics,
        "test_metrics": test_result.metrics,
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    model_dir = out_dir / "final_model"
    trainer.save_model(model_dir)
    processor.save_pretrained(model_dir)

    print("\n" + "=" * 70)
    print("FINAL RESULTS ON OFFICIAL TEST SET")
    print("=" * 70)
    print(
        f"Test CER: {test_result.metrics['test_cer']:.4f} ({test_result.metrics['test_cer'] * 100:.2f}%)"
    )
    print(
        f"Test WER: {test_result.metrics['test_wer']:.4f} ({test_result.metrics['test_wer'] * 100:.2f}%)"
    )
    print(f"\nSaved to: {out_dir}")
    print("=" * 70)

    BOZEN_CER = test_result.metrics["test_cer"]
    BOZEN_WER = test_result.metrics["test_wer"]

    print("\n✅ TRAINING COMPLETE!")

except Exception as e:
    print(f"\n❌ TRAINING FAILED")
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
