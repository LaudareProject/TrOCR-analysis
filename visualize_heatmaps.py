#!/usr/bin/env python3
"""
Script to colorize grayscale attention maps and GradCAM visualizations as heatmaps,
overlay them on the original image with specified opacity, and stretch to desired aspect ratio.

Usage:
    python visualize_heatmaps.py <sample_number> <aspect_ratio>
    
Example:
    python visualize_heatmaps.py 0020 11
"""

import os
import sys
import argparse
import glob
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Tuple, List


def load_image(image_path: str) -> np.ndarray:
    """Load an image and convert to numpy array."""
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def load_grayscale_image(image_path: str) -> np.ndarray:
    """Load a grayscale image and return as numpy array."""
    try:
        img = Image.open(image_path)
        if img.mode != 'L':
            img = img.convert('L')
        return np.array(img)
    except Exception as e:
        print(f"Error loading grayscale image {image_path}: {e}")
        return None


def colorize_heatmap(grayscale_array: np.ndarray, colormap: str = 'jet') -> np.ndarray:
    """
    Convert grayscale array to colored heatmap using matplotlib colormap.
    
    Args:
        grayscale_array: 2D numpy array with grayscale values
        colormap: matplotlib colormap name (e.g., 'jet', 'hot', 'viridis')
    
    Returns:
        3D numpy array (H, W, 3) representing RGB heatmap
    """
    # Normalize to 0-1 range
    normalized = grayscale_array.astype(np.float32) / 255.0
    
    # Apply colormap
    cmap = plt.colormaps[colormap]
    colored = cmap(normalized)
    
    # Convert to 0-255 RGB (remove alpha channel)
    rgb_heatmap = (colored[:, :, :3] * 255).astype(np.uint8)
    
    return rgb_heatmap


def resize_to_match(source_img: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Resize source image to match target shape."""
    source_pil = Image.fromarray(source_img)
    resized = source_pil.resize((target_shape[1], target_shape[0]), Image.LANCZOS)
    return np.array(resized)


def overlay_images(background: np.ndarray, overlay: np.ndarray, opacity: float = 0.5) -> np.ndarray:
    """
    Overlay two images with specified opacity.
    
    Args:
        background: Background image (RGB)
        overlay: Overlay image (RGB)
        opacity: Opacity of overlay (0.0 to 1.0)
    
    Returns:
        Combined image
    """
    # Ensure both images are float for blending
    bg_float = background.astype(np.float32)
    overlay_float = overlay.astype(np.float32)
    
    # Blend images
    blended = (1 - opacity) * bg_float + opacity * overlay_float
    
    # Convert back to uint8
    return np.clip(blended, 0, 255).astype(np.uint8)


def stretch_image(image: np.ndarray, target_aspect_ratio: float) -> np.ndarray:
    """
    Stretch image to achieve target width/height aspect ratio.
    
    Args:
        image: Input image array
        target_aspect_ratio: Desired width/height ratio
    
    Returns:
        Stretched image
    """
    height, width = image.shape[:2]
    current_ratio = width / height
    
    if current_ratio < target_aspect_ratio:
        # Need to stretch width
        new_width = int(height * target_aspect_ratio)
        new_height = height
    else:
        # Need to stretch height  
        new_width = width
        new_height = int(width / target_aspect_ratio)
    
    # Resize image
    pil_img = Image.fromarray(image)
    stretched = pil_img.resize((new_width, new_height), Image.LANCZOS)
    return np.array(stretched)


def process_sample(sample_number: str, aspect_ratio: float, opacity: float = 0.5):
    """
    Process all visualizations for a given sample.
    
    Args:
        sample_number: Sample number (e.g., '0020')
        aspect_ratio: Target width/height ratio
        opacity: Overlay opacity
    """
    # Define paths
    base_path = Path("combined_visualizations") / f"sample_{sample_number}" / "raw_maps"
    output_path = Path("combined_visualizations") / f"sample_{sample_number}" / "processed_heatmaps"
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load original image
    input_image_path = base_path / "input_image.png"
    if not input_image_path.exists():
        print(f"Error: Input image not found at {input_image_path}")
        return
    
    original_img = load_image(str(input_image_path))
    if original_img is None:
        return
    
    print(f"Processing sample {sample_number}...")
    print(f"Original image shape: {original_img.shape}")
    
    # Save the stretched original image (barebone PNG)
    stretched_original = stretch_image(original_img, aspect_ratio)
    original_output_path = output_path / f"barebone_stretched.png"
    Image.fromarray(stretched_original).save(original_output_path)
    print(f"Saved barebone image: {original_output_path}")
    
    # Find all attention and gradcam files
    attention_files = list(base_path.glob("*_attention.png"))
    gradcam_files = list(base_path.glob("*_gradcam.png"))
    
    print(f"Found {len(attention_files)} attention maps and {len(gradcam_files)} gradcam maps")
    
    # Process attention maps
    for attention_file in attention_files:
        process_visualization_file(
            attention_file, original_img, output_path, 
            aspect_ratio, opacity, "attention", "hot"
        )
    
    # Process gradcam maps
    for gradcam_file in gradcam_files:
        process_visualization_file(
            gradcam_file, original_img, output_path,
            aspect_ratio, opacity, "gradcam", "jet"
        )
    
    # Create a combined full heatmap by averaging all attention and gradcam maps
    create_full_heatmap(attention_files + gradcam_files, original_img, output_path, aspect_ratio, opacity)
    
    print(f"Processing complete! Results saved to {output_path}")


def process_visualization_file(vis_file: Path, original_img: np.ndarray, 
                             output_path: Path, aspect_ratio: float, 
                             opacity: float, vis_type: str, colormap: str):
    """Process a single visualization file."""
    # Load the grayscale visualization
    vis_gray = load_grayscale_image(str(vis_file))
    if vis_gray is None:
        return
    
    # Colorize the heatmap
    vis_colored = colorize_heatmap(vis_gray, colormap)
    
    # Resize to match original image dimensions
    vis_resized = resize_to_match(vis_colored, original_img.shape[:2])
    
    # Overlay on original image
    overlaid = overlay_images(original_img, vis_resized, opacity)
    
    # Stretch to target aspect ratio
    stretched = stretch_image(overlaid, aspect_ratio)
    
    # Generate output filename
    stem = vis_file.stem.replace('_attention', '').replace('_gradcam', '')
    output_filename = f"{stem}_{vis_type}_overlay_stretched.png"
    output_file_path = output_path / output_filename
    
    # Save result
    Image.fromarray(stretched).save(output_file_path)
    print(f"Saved: {output_file_path}")


def create_full_heatmap(vis_files: List[Path], original_img: np.ndarray, 
                       output_path: Path, aspect_ratio: float, opacity: float):
    """Create a combined full heatmap from all visualization files."""
    if not vis_files:
        return
    
    print("Creating full combined heatmap...")
    
    # Initialize accumulator for the combined heatmap
    combined_heatmap = None
    valid_files = 0
    
    for vis_file in vis_files:
        # Load the grayscale visualization
        vis_gray = load_grayscale_image(str(vis_file))
        if vis_gray is None:
            continue
        
        # Resize to match original image dimensions
        vis_resized = resize_to_match(vis_gray, original_img.shape[:2])
        
        # Normalize to 0-1 range
        vis_normalized = vis_resized.astype(np.float32) / 255.0
        
        # Add to combined heatmap
        if combined_heatmap is None:
            combined_heatmap = vis_normalized
        else:
            combined_heatmap += vis_normalized
        
        valid_files += 1
    
    if combined_heatmap is None or valid_files == 0:
        print("No valid visualization files found for full heatmap")
        return
    
    # Average the combined heatmap
    combined_heatmap = combined_heatmap / valid_files
    
    # Convert back to 0-255 range
    combined_gray = (combined_heatmap * 255).astype(np.uint8)
    
    # Colorize using a vibrant colormap for the full heatmap
    combined_colored = colorize_heatmap(combined_gray, 'plasma')
    
    # Overlay on original image
    overlaid = overlay_images(original_img, combined_colored, opacity)
    
    # Stretch to target aspect ratio
    stretched = stretch_image(overlaid, aspect_ratio)
    
    # Save the full heatmap
    full_heatmap_path = output_path / "full_heatmap_combined.png"
    Image.fromarray(stretched).save(full_heatmap_path)
    print(f"Saved full combined heatmap: {full_heatmap_path}")
    print(f"Combined {valid_files} visualization maps")


def main():
    parser = argparse.ArgumentParser(
        description="Colorize attention maps and overlay on original images"
    )
    parser.add_argument(
        "sample_number", 
        help="Sample number (e.g., 0020)"
    )
    parser.add_argument(
        "aspect_ratio", 
        type=float,
        help="Target width/height aspect ratio (e.g., 11)"
    )
    parser.add_argument(
        "--opacity", 
        type=float, 
        default=0.5,
        help="Overlay opacity (0.0 to 1.0, default: 0.5)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.aspect_ratio <= 0:
        print("Error: Aspect ratio must be positive")
        sys.exit(1)
    
    if not (0.0 <= args.opacity <= 1.0):
        print("Error: Opacity must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Process the sample
    process_sample(args.sample_number, args.aspect_ratio, args.opacity)


if __name__ == "__main__":
    main()