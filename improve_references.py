#!/usr/bin/env python3
"""
Script to improve reference images with aesthetic filtering and regenerate visualizations.
This script loads existing processing results and only re-extracts/enhances reference images.
"""

import cv2
import numpy as np
import pickle
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import argparse


def calculate_aesthetic_score(image):
    """
    Calculate aesthetic score for an image based on multiple factors:
    - Sharpness (Laplacian variance)
    - Brightness/exposure
    - Contrast
    - Color vibrancy
    - Face centering
    - Symmetry
    """
    if len(image.shape) == 2:
        gray = image
        color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        color = image
    
    h, w = gray.shape
    
    # 1. Sharpness (Laplacian variance)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(lap_var / 1000.0, 1.0)
    
    # 2. Brightness - prefer well-lit faces (not too dark, not overexposed)
    brightness = gray.mean()
    # Optimal brightness around 120-140
    brightness_score = 1.0 - abs(brightness - 130) / 130.0
    brightness_score = max(0, brightness_score)
    
    # 3. Contrast
    contrast = gray.std()
    contrast_score = min(contrast / 70.0, 1.0)
    
    # 4. Color vibrancy (for color images)
    if len(image.shape) == 3:
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].mean()
        vibrancy_score = min(saturation / 128.0, 1.0)
    else:
        vibrancy_score = 0.5
    
    # 5. Exposure check - penalize overexposed or underexposed areas
    overexposed = np.sum(gray > 240) / gray.size
    underexposed = np.sum(gray < 20) / gray.size
    exposure_score = 1.0 - (overexposed + underexposed)
    
    # 6. Edge definition
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size
    edge_score = min(edge_density / 0.15, 1.0)
    
    # Weighted combination
    aesthetic_score = (
        sharpness_score * 0.25 +
        brightness_score * 0.20 +
        contrast_score * 0.15 +
        vibrancy_score * 0.15 +
        exposure_score * 0.15 +
        edge_score * 0.10
    )
    
    return aesthetic_score


def enhance_reference_image(image):
    """
    Apply aesthetic enhancements to reference image:
    - Sharpening
    - Color correction
    - Contrast enhancement
    - Slight saturation boost
    """
    # Convert to PIL for better enhancement controls
    if len(image.shape) == 2:
        pil_image = Image.fromarray(image)
    else:
        # OpenCV uses BGR, PIL uses RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
    
    # 1. Sharpen
    pil_image = pil_image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    
    # 2. Enhance contrast slightly
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.15)
    
    # 3. Enhance color/saturation
    enhancer = ImageEnhance.Color(pil_image)
    pil_image = enhancer.enhance(1.1)
    
    # 4. Adjust brightness if needed
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(1.05)
    
    # Convert back to numpy array
    enhanced = np.array(pil_image)
    
    # Convert back to BGR if it was color
    if len(image.shape) == 3:
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
    
    return enhanced


def extract_improved_reference_images(video_path, reference_faces, output_dir, num_candidates=10):
    """
    Extract improved reference images by:
    1. Finding multiple candidate frames for each character
    2. Scoring them aesthetically
    3. Picking the best one
    4. Applying enhancement filters
    """
    print(f"Extracting improved reference images from video...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return {}
    
    reference_images = {}
    
    character_frames = {}
    for char_id, face_data in reference_faces.items():
        character_frames[char_id] = [{
            'face_id': char_id,
            'frame_idx': face_data['frame_idx'],
            'bbox': face_data['bbox'],
            'quality': face_data.get('confidence', 0.5)
        }]
    
    print(f"Found {len(character_frames)} unique characters")
    
    # For each character, find best reference frame
    for char_id in tqdm(sorted(character_frames.keys()), desc="Processing characters"):
        faces = character_frames[char_id]
        
        # Sort by quality and take top candidates
        faces_sorted = sorted(faces, key=lambda x: x['quality'], reverse=True)
        candidates = faces_sorted[:min(num_candidates, len(faces_sorted))]
        
        best_frame = None
        best_aesthetic_score = -1
        best_face_crop = None
        
        # Evaluate each candidate
        for candidate in candidates:
            frame_idx = candidate['frame_idx']
            bbox = candidate['bbox']
            
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret or frame is None:
                continue
            
            # Extract face region
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            # Add padding around face
            padding = 0.3
            w, h = x2 - x1, y2 - y1
            pad_w, pad_h = int(w * padding), int(h * padding)
            
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(frame.shape[1], x2 + pad_w)
            y2 = min(frame.shape[0], y2 + pad_h)
            
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                continue
            
            # Calculate aesthetic score
            aesthetic_score = calculate_aesthetic_score(face_crop)
            
            # Combine with original quality score
            combined_score = aesthetic_score * 0.7 + candidate['quality'] * 0.3
            
            if combined_score > best_aesthetic_score:
                best_aesthetic_score = combined_score
                best_frame = frame_idx
                best_face_crop = face_crop.copy()
        
        if best_face_crop is not None:
            # Enhance the best reference image
            enhanced = enhance_reference_image(best_face_crop)
            
            # Resize to standard size for consistency
            target_size = (224, 224)
            enhanced_resized = cv2.resize(enhanced, target_size, interpolation=cv2.INTER_LANCZOS4)
            
            reference_images[char_id] = enhanced_resized
            
            # Save individual reference image
            ref_dir = os.path.join(output_dir, 'reference_images')
            os.makedirs(ref_dir, exist_ok=True)
            cv2.imwrite(
                os.path.join(ref_dir, f'character_{char_id}_ref.jpg'),
                enhanced_resized,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
    
    cap.release()
    
    print(f"Extracted {len(reference_images)} improved reference images")
    return reference_images


def create_improved_visualization(frame, matches, reference_images, frame_idx):
    """
    Create visualization with improved reference images.
    """
    fig, axes = plt.subplots(1, len(matches) + 1, figsize=(4 * (len(matches) + 1), 5))
    
    if len(matches) == 0:
        axes = [axes]
    
    # Display input frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    axes[0].imshow(frame_rgb)
    axes[0].set_title('Input Frame', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Draw bounding boxes on input frame
    for match in matches:
        bbox = match['bbox']
        rect = plt.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            fill=False,
            edgecolor='lime',
            linewidth=2
        )
        axes[0].add_patch(rect)
        
        # Add character label
        axes[0].text(
            bbox[0],
            bbox[1] - 5,
            f"Char {match['character_id']}",
            color='lime',
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7)
        )
    
    # Display matched reference faces
    for idx, match in enumerate(matches):
        char_id = match['character_id']
        
        if char_id in reference_images:
            ref_img = reference_images[char_id]
            ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
            axes[idx + 1].imshow(ref_rgb)
            axes[idx + 1].set_title(
                f'Character {char_id}',
                fontsize=14,
                fontweight='bold'
            )
            axes[idx + 1].axis('off')
            
            # Add similarity score
            axes[idx + 1].text(
                0.5, -0.05,
                f'Similarity: {match["similarity"]:.3f}',
                transform=axes[idx + 1].transAxes,
                ha='center',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
            )
    
    plt.tight_layout()
    return fig


def regenerate_visualizations(output_dir, reference_images):
    """
    Regenerate all visualization images with improved references.
    """
    print("\nRegenerating visualizations with improved references...")
    
    # Load frame matches
    matches_file = os.path.join(output_dir, 'frame_matches.json')
    if not os.path.exists(matches_file):
        print(f"Error: frame_matches.json not found in {output_dir}")
        return False
    
    with open(matches_file, 'r') as f:
        frame_matches = json.load(f)
    
    # Get video path from log or reconstruct
    video_name = '_'.join(os.path.basename(output_dir).split('_')[:-2])
    
    # Try to find video file
    videos_base = '/mnt/data/data_hub/movies_dataset/videos'
    video_path = None
    
    for root, dirs, files in os.walk(videos_base):
        for file in files:
            if video_name in file and file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(root, file)
                break
        if video_path:
            break
    
    if not video_path or not os.path.exists(video_path):
        print(f"Error: Could not find video file for {video_name}")
        return False
    
    print(f"Using video: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
    
    # Create new visualizations directory
    viz_dir = os.path.join(output_dir, 'visualizations_improved')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Process each frame
    for frame_data in tqdm(frame_matches, desc="Regenerating visualizations"):
        frame_idx = frame_data['frame_idx']
        matches = frame_data['matches']
        
        # Read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret or frame is None:
            continue
        
        # Create visualization
        fig = create_improved_visualization(frame, matches, reference_images, frame_idx)
        
        # Save
        output_path = os.path.join(viz_dir, f'frame_{int(frame_idx):06d}_viz.jpg')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    cap.release()
    
    # Optionally backup old visualizations and replace
    old_viz_dir = os.path.join(output_dir, 'visualizations')
    if os.path.exists(old_viz_dir):
        backup_dir = os.path.join(output_dir, 'visualizations_old')
        if os.path.exists(backup_dir):
            import shutil
            shutil.rmtree(backup_dir)
        os.rename(old_viz_dir, backup_dir)
        print(f"Backed up old visualizations to: visualizations_old/")
    
    os.rename(viz_dir, old_viz_dir)
    print(f"✓ Regenerated visualizations saved to: visualizations/")
    
    return True


def process_directory(output_dir, num_candidates=10):
    """
    Process a single output directory to improve references and regenerate visualizations.
    """
    print(f"\n{'='*80}")
    print(f"Processing: {os.path.basename(output_dir)}")
    print(f"{'='*80}")
    
    # Check if required files exist
    ref_faces_file = os.path.join(output_dir, 'reference_faces.pkl')
    if not os.path.exists(ref_faces_file):
        print(f"Error: reference_faces.pkl not found in {output_dir}")
        return False
    
    # Load reference faces
    print("Loading reference faces...")
    with open(ref_faces_file, 'rb') as f:
        reference_faces = pickle.load(f)
    
    print(f"Loaded {len(reference_faces)} reference faces")
    
    # Get video path
    video_name = '_'.join(os.path.basename(output_dir).split('_')[:-2])
    videos_base = '/mnt/data/data_hub/movies_dataset/videos'
    video_path = None
    
    for root, dirs, files in os.walk(videos_base):
        for file in files:
            if video_name in file and file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(root, file)
                break
        if video_path:
            break
    
    if not video_path or not os.path.exists(video_path):
        print(f"Error: Could not find video file for {video_name}")
        return False
    
    print(f"Found video: {video_path}")
    
    # Extract improved reference images
    reference_images = extract_improved_reference_images(
        video_path, reference_faces, output_dir, num_candidates
    )
    
    if not reference_images:
        print("Error: Failed to extract improved reference images")
        return False
    
    # Regenerate visualizations
    success = regenerate_visualizations(output_dir, reference_images)
    
    if success:
        print(f"✓ Successfully improved references and regenerated visualizations")
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description='Improve reference images and regenerate visualizations'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory to process (single directory or parent directory)'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process all subdirectories in the output-dir'
    )
    parser.add_argument(
        '--num-candidates',
        type=int,
        default=10,
        help='Number of candidate frames to evaluate per character (default: 10)'
    )
    parser.add_argument(
        '--filter',
        type=str,
        help='Only process directories matching this string'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        print(f"Error: Directory not found: {args.output_dir}")
        sys.exit(1)
    
    if args.batch:
        # Process multiple directories
        print(f"Batch processing mode - scanning {args.output_dir}")
        
        dirs_to_process = []
        for item in os.listdir(args.output_dir):
            if item == 'archive_facemap':
                continue
            
            item_path = os.path.join(args.output_dir, item)
            if not os.path.isdir(item_path):
                continue
            
            # Check if it has the required files
            if not os.path.exists(os.path.join(item_path, 'reference_faces.pkl')):
                continue
            
            # Apply filter if specified
            if args.filter and args.filter.lower() not in item.lower():
                continue
            
            dirs_to_process.append(item_path)
        
        print(f"Found {len(dirs_to_process)} directories to process")
        
        successful = 0
        failed = 0
        
        for dir_path in dirs_to_process:
            try:
                if process_directory(dir_path, args.num_candidates):
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Error processing {dir_path}: {str(e)}")
                failed += 1
        
        print(f"\n{'='*80}")
        print(f"Batch processing complete")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"{'='*80}")
    else:
        # Process single directory
        process_directory(args.output_dir, args.num_candidates)


if __name__ == "__main__":
    main()
