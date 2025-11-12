import argparse
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import pickle
import os
import json
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import logging
import sys
import traceback
import ray
import time
import shutil

warnings.filterwarnings('ignore')


def setup_logging(output_dir):
    log_file = os.path.join(output_dir, 'processing.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def get_video_files(videos_dir, max_folders=None):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    video_files = []
    
    subdirs = sorted([d for d in os.listdir(videos_dir) if os.path.isdir(os.path.join(videos_dir, d))])
    
    if max_folders:
        subdirs = subdirs[:max_folders]
    
    for subdir in subdirs:
        subdir_path = os.path.join(videos_dir, subdir)
        for root, dirs, files in os.walk(subdir_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_files.append(os.path.join(root, file))
    
    return video_files


def calculate_multi_scale_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    h, w = gray.shape
    
    if h < 32 or w < 32:
        return 0.0
    
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness = min(lap_var / 500.0, 1.0)
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = np.sqrt(sobelx**2 + sobely**2).mean()
    edge_score = min(edge_strength / 50.0, 1.0)
    
    brightness = gray.mean()
    brightness_score = 1.0 - abs(brightness - 127.5) / 127.5
    
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()
    entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-7))
    entropy_score = min(entropy / 8.0, 1.0)
    
    contrast = gray.std()
    contrast_score = min(contrast / 64.0, 1.0)
    
    quality_score = (sharpness * 0.35 + edge_score * 0.25 + brightness_score * 0.15 + 
                     entropy_score * 0.15 + contrast_score * 0.10)
    
    return quality_score


def calculate_aesthetic_score(image):
    if len(image.shape) == 2:
        gray = image
        color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        color = image
    
    h, w = gray.shape
    
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(lap_var / 1000.0, 1.0)
    
    brightness = gray.mean()
    brightness_score = 1.0 - abs(brightness - 130) / 130.0
    brightness_score = max(0, brightness_score)
    
    contrast = gray.std()
    contrast_score = min(contrast / 70.0, 1.0)
    
    if len(image.shape) == 3:
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].mean()
        vibrancy_score = min(saturation / 128.0, 1.0)
    else:
        vibrancy_score = 0.5
    
    overexposed = np.sum(gray > 240) / gray.size
    underexposed = np.sum(gray < 20) / gray.size
    exposure_score = 1.0 - (overexposed + underexposed)
    
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size
    edge_score = min(edge_density / 0.15, 1.0)
    
    aesthetic_score = (
        sharpness_score * 0.25 +
        brightness_score * 0.20 +
        contrast_score * 0.15 +
        vibrancy_score * 0.15 +
        exposure_score * 0.15 +
        edge_score * 0.10
    )
    
    return aesthetic_score


def enhance_image(image):
    if len(image.shape) == 2:
        pil_image = Image.fromarray(image)
    else:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
    
    pil_image = pil_image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.15)
    
    enhancer = ImageEnhance.Color(pil_image)
    pil_image = enhancer.enhance(1.1)
    
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(1.05)
    
    enhanced = np.array(pil_image)
    
    if len(image.shape) == 3:
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
    
    return enhanced


def extract_faces_from_video(video_path, sample_rate, min_face_size, app, logger):
    logger.info(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video properties: {total_frames} frames, {fps:.2f} FPS, {width}x{height}")
    
    all_faces = []
    frame_indices = []
    face_boxes = []
    
    frame_idx = 0
    frames_to_process = total_frames // sample_rate
    
    pbar = tqdm(total=frames_to_process, desc="Extracting faces", unit="frame", leave=False)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_rate == 0:
            faces = app.get(frame)
            
            for face in faces:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                if (x2 - x1) >= min_face_size and (y2 - y1) >= min_face_size:
                    face_crop = frame[y1:y2, x1:x2]
                    
                    if face_crop.size == 0:
                        continue
                    
                    quality_score = calculate_multi_scale_quality(face_crop)
                    aesthetic_score = calculate_aesthetic_score(face_crop)
                    
                    all_faces.append(face.embedding)
                    frame_indices.append(frame_idx)
                    face_boxes.append({
                        'frame_idx': int(frame_idx),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(face.det_score),
                        'quality_score': float(quality_score),
                        'aesthetic_score': float(aesthetic_score)
                    })
            
            pbar.update(1)
        
        frame_idx += 1
    
    cap.release()
    pbar.close()
    
    logger.info(f"Extraction complete: {len(all_faces)} faces from {frame_idx} frames")
    
    return np.array(all_faces), frame_indices, face_boxes


def cluster_faces(embeddings, face_boxes, logger, eps=0.35, min_samples=5):
    logger.info(f"Clustering {len(embeddings)} faces")
    
    similarity_matrix = cosine_similarity(embeddings)
    distance_matrix = 1 - similarity_matrix
    distance_matrix = np.clip(distance_matrix, 0, None)
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = clustering.fit_predict(distance_matrix)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    logger.info(f"Clustering complete: {n_clusters} characters, {n_noise} noise points")
    
    return labels


def select_best_reference_faces(embeddings, labels, face_boxes, video_path, num_candidates=10):
    reference_faces = {}
    
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    for label in tqdm(unique_labels, desc="Selecting references", leave=False):
        cluster_indices = np.where(labels == label)[0]
        cluster_boxes = [face_boxes[i] for i in cluster_indices]
        
        quality_scores = []
        for box in cluster_boxes:
            base_score = (box['confidence'] * 0.3 + 
                         box['quality_score'] * 0.3 + 
                         box['aesthetic_score'] * 0.4)
            quality_scores.append(base_score)
        
        top_candidates_idx = np.argsort(quality_scores)[-num_candidates:]
        top_candidates = [cluster_boxes[i] for i in top_candidates_idx]
        
        best_frame = None
        best_aesthetic = -1
        best_box = None
        
        for candidate in top_candidates:
            frame_idx = candidate['frame_idx']
            bbox = candidate['bbox']
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret or frame is None:
                continue
            
            x1, y1, x2, y2 = bbox
            
            padding = 0.3
            w, h = x2 - x1, y2 - y1
            pad_w, pad_h = int(w * padding), int(h * padding)
            
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(width, x2 + pad_w)
            y2 = min(height, y2 + pad_h)
            
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                continue
            
            aesthetic_score = calculate_aesthetic_score(face_crop)
            combined_score = aesthetic_score * 0.7 + candidate['quality_score'] * 0.3
            
            if combined_score > best_aesthetic:
                best_aesthetic = combined_score
                best_frame = frame_idx
                best_box = bbox
        
        if best_box is not None:
            best_idx = cluster_indices[np.argmax(quality_scores)]
            
            reference_faces[f"character_{label}"] = {
                'embedding': embeddings[best_idx],
                'frame_idx': best_frame,
                'bbox': best_box,
                'confidence': face_boxes[best_idx]['confidence'],
                'quality_score': face_boxes[best_idx]['quality_score'],
                'aesthetic_score': best_aesthetic,
                'cluster_size': len(cluster_indices)
            }
    
    cap.release()
    return reference_faces


def extract_reference_images(video_path, reference_faces, output_dir, logger):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ref_images = {}
    
    logger.info("Extracting and enhancing reference images")
    
    for char_id, data in tqdm(reference_faces.items(), desc="Extracting references", leave=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, data['frame_idx'])
        ret, frame = cap.read()
        
        if ret:
            x1, y1, x2, y2 = data['bbox']
            
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                continue
            
            enhanced = enhance_image(face_crop)
            enhanced_resized = cv2.resize(enhanced, (512, 512), interpolation=cv2.INTER_LANCZOS4)
            
            face_rgb = cv2.cvtColor(enhanced_resized, cv2.COLOR_BGR2RGB)
            
            ref_path = f"{output_dir}/{char_id}_reference.jpg"
            Image.fromarray(face_rgb).save(ref_path, quality=95)
            ref_images[char_id] = face_rgb
    
    cap.release()
    logger.info(f"Extracted {len(ref_images)} enhanced reference images")
    return ref_images


def generate_test_frames_from_video(video_path, num_frames, output_dir, logger):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    test_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    test_frames_dir = f"{output_dir}/test_frames"
    os.makedirs(test_frames_dir, exist_ok=True)
    
    logger.info(f"Generating {num_frames} test frames")
    saved_frames = []
    
    for idx in tqdm(test_indices, desc="Generating test frames", leave=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_path = f"{test_frames_dir}/frame_{idx:06d}.jpg"
            cv2.imwrite(frame_path, frame)
            saved_frames.append((int(idx), frame_path, frame))
    
    cap.release()
    logger.info(f"Generated {len(saved_frames)} test frames")
    return saved_frames


def match_faces_in_frame(frame, reference_faces, similarity_threshold, app):
    faces = app.get(frame)
    
    matched_faces = []
    
    for face in faces:
        bbox = face.bbox.astype(int)
        embedding = face.embedding
        
        best_match = None
        best_similarity = -1
        
        for char_id, ref_data in reference_faces.items():
            ref_embedding = ref_data['embedding']
            similarity = cosine_similarity([embedding], [ref_embedding])[0][0]
            
            if similarity > best_similarity and similarity > similarity_threshold:
                best_similarity = similarity
                best_match = char_id
        
        matched_faces.append({
            'bbox': bbox.tolist(),
            'character_id': best_match,
            'confidence': float(face.det_score),
            'similarity': float(best_similarity) if best_match else 0.0
        })
    
    return matched_faces


def create_visualization(frame, matches, reference_images, frame_idx):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    detected_chars = [m['character_id'] for m in matches if m['character_id']]
    
    if len(detected_chars) == 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.imshow(frame_rgb)
        
        for match in matches:
            bbox = match['bbox']
            rect = plt.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                fill=False,
                edgecolor='red',
                linewidth=2
            )
            ax.add_patch(rect)
        
        ax.set_title(f'Frame {frame_idx}', fontsize=12)
        ax.axis('off')
        return fig
    
    cols = len(detected_chars) + 1
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 5))
    
    if cols == 2:
        axes = [axes[0], axes[1]]
    
    axes[0].imshow(frame_rgb)
    axes[0].set_title('Input Frame', fontsize=12)
    axes[0].axis('off')
    
    for match in matches:
        if match['character_id']:
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
    
    for idx, char_id in enumerate(detected_chars, 1):
        if char_id in reference_images:
            axes[idx].imshow(reference_images[char_id])
            axes[idx].set_title(char_id.replace('character_', 'Character '), fontsize=12)
            axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def generate_summary_report(frame_matches, reference_faces):
    total_frames = len(frame_matches)
    frames_with_faces = sum(1 for fm in frame_matches if len(fm['matches']) > 0)
    frames_no_faces = total_frames - frames_with_faces
    
    frames_with_matches = sum(1 for fm in frame_matches 
                              if any(m['character_id'] for m in fm['matches']))
    
    char_appearances = defaultdict(int)
    for fm in frame_matches:
        detected_chars = set()
        for match in fm['matches']:
            if match['character_id']:
                detected_chars.add(match['character_id'])
        for char in detected_chars:
            char_appearances[char] += 1
    
    report = f"""
FACE MATCHING SUMMARY REPORT

VIDEO ANALYSIS:
- Total test frames processed: {total_frames}
- Frames with detected faces: {frames_with_faces} ({frames_with_faces/total_frames*100:.1f}%)
- Frames with matched characters: {frames_with_matches} ({frames_with_matches/total_frames*100:.1f}%)
- Frames with no faces: {frames_no_faces} ({frames_no_faces/total_frames*100:.1f}%)

CHARACTER REFERENCE DATABASE:
- Total unique characters: {len(reference_faces)}

CHARACTER APPEARANCES IN TEST FRAMES:
"""
    
    for char_id in sorted(char_appearances.keys()):
        count = char_appearances[char_id]
        percentage = count / total_frames * 100
        report += f"- {char_id}: {count} frames ({percentage:.1f}%)\n"
    
    return report


@ray.remote(num_gpus=0.0625)
def process_single_video(video_path, output_root, args):
    try:
        video_name = Path(video_path).stem
        
        existing_dirs = [d for d in os.listdir(output_root) if d.startswith(video_name + "_")]
        
        archive_dir = os.path.join(output_root, "archive_facemap")
        
        if existing_dirs:
            os.makedirs(archive_dir, exist_ok=True)
            for old_dir in existing_dirs:
                old_path = os.path.join(output_root, old_dir)
                archive_path = os.path.join(archive_dir, old_dir)
                if os.path.exists(old_path):
                    shutil.move(old_path, archive_path)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_root, f"{video_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        logger = setup_logging(output_dir)
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Output directory: {output_dir}")
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        app = FaceAnalysis(
            name='buffalo_sc',
            providers=providers
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        logger.info("Model loaded: buffalo_sc")
        
        embeddings, frame_indices, face_boxes = extract_faces_from_video(
            video_path, args['sample_rate'], args['min_face_size'], app, logger
        )
        
        if len(embeddings) == 0:
            logger.warning("No faces found in video")
            return False
        
        labels = cluster_faces(embeddings, face_boxes, logger)
        
        reference_faces = select_best_reference_faces(
            embeddings, labels, face_boxes, video_path, num_candidates=args['num_candidates']
        )
        logger.info(f"Selected {len(reference_faces)} reference faces")
        
        reference_images = extract_reference_images(video_path, reference_faces, output_dir, logger)
        
        test_frames = generate_test_frames_from_video(video_path, args['num_test_frames'], output_dir, logger)
        
        viz_dir = f"{output_dir}/visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        logger.info("Matching faces and creating visualizations")
        frame_matches = []
        
        for frame_idx, frame_path, frame in tqdm(test_frames, desc="Processing test frames", leave=False):
            matches = match_faces_in_frame(frame, reference_faces, args['similarity_threshold'], app)
            
            frame_matches.append({
                'frame_idx': int(frame_idx),
                'frame_path': frame_path,
                'matches': matches
            })
            
            fig = create_visualization(frame, matches, reference_images, frame_idx)
            output_path = f"{viz_dir}/frame_{int(frame_idx):06d}_viz.jpg"
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        matches_output = []
        for fm in frame_matches:
            matches_output.append({
                'frame_idx': int(fm['frame_idx']),
                'matches': fm['matches']
            })
        
        with open(f"{output_dir}/frame_matches.json", 'w') as f:
            json.dump(matches_output, f, indent=2)
        
        with open(f"{output_dir}/reference_faces.pkl", 'wb') as f:
            pickle.dump(reference_faces, f)
        
        summary_report = generate_summary_report(frame_matches, reference_faces)
        
        with open(f"{output_dir}/summary_report.txt", 'w') as f:
            f.write(summary_report)
        
        logger.info(summary_report)
        logger.info(f"Processing complete for {video_name}")
        
        return True
        
    except Exception as e:
        error_msg = f"Error processing {video_path}: {str(e)}\n{traceback.format_exc()}"
        if 'logger' in locals():
            logger.error(error_msg)
        else:
            print(error_msg)
        return False


def process_videos_batch(video_files, output_root, args):
    total_videos = len(video_files)
    
    print(f"\nStarting batch processing")
    print(f"Total videos: {total_videos}")
    print(f"Model: buffalo_sc")
    print(f"GPUs: 2")
    print(f"Concurrent tasks: 32")
    print(f"Candidate frames per character: {args.num_candidates}")
    print(f"Test frames per video: {args.num_test_frames}")
    
    ray.init(num_gpus=2, num_cpus=32)
    
    args_dict = {
        'num_test_frames': args.num_test_frames,
        'sample_rate': args.sample_rate,
        'min_face_size': args.min_face_size,
        'similarity_threshold': args.similarity_threshold,
        'num_candidates': args.num_candidates
    }
    
    futures = [process_single_video.remote(video_path, output_root, args_dict) for video_path in video_files]
    
    results = []
    start_time = time.time()
    
    with tqdm(total=total_videos, desc="Overall progress", unit="video") as pbar:
        while len(futures) > 0:
            done, futures = ray.wait(futures, timeout=1.0)
            for future in done:
                results.append(ray.get(future))
                pbar.update(1)
                
                completed = len(results)
                elapsed = time.time() - start_time
                if completed > 0:
                    avg_time_per_video = elapsed / completed
                    remaining = total_videos - completed
                    eta_seconds = avg_time_per_video * remaining
                    eta_hours = int(eta_seconds // 3600)
                    eta_minutes = int((eta_seconds % 3600) // 60)
                    pbar.set_postfix({
                        'Success': sum(results),
                        'Failed': completed - sum(results),
                        'ETA': f'{eta_hours}h {eta_minutes}m'
                    })
    
    ray.shutdown()
    
    successful = sum(results)
    failed = len(results) - successful
    total_time = time.time() - start_time
    
    print(f"\nBatch processing complete")
    print(f"Total videos processed: {total_videos}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(successful/total_videos)*100:.1f}%")
    print(f"Total time: {int(total_time//3600)}h {int((total_time%3600)//60)}m")
    print(f"Output directory: {output_root}")


def main():
    parser = argparse.ArgumentParser(description='Optimized Face Recognition Pipeline for Movies')
    
    parser.add_argument('--videos-dir', type=str, default='/mnt/data/data_hub/movies_dataset/videos',
                        help='Directory containing video files')
    parser.add_argument('--output-dir', type=str, default='/mnt/data/data_hub/movies_dataset/face_extraction',
                        help='Root output directory')
    parser.add_argument('--num-test-frames', type=int, default=150,
                        help='Number of test frames to generate per video')
    parser.add_argument('--sample-rate', type=int, default=30,
                        help='Frame sampling rate for face extraction')
    parser.add_argument('--min-face-size', type=int, default=30,
                        help='Minimum face size in pixels')
    parser.add_argument('--similarity-threshold', type=float, default=0.35,
                        help='Similarity threshold for face matching')
    parser.add_argument('--max-folders', type=int, default=None,
                        help='Maximum number of folders to process')
    parser.add_argument('--num-candidates', type=int, default=15,
                        help='Number of candidate frames to evaluate per character')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.videos_dir):
        print(f"Error: Videos directory not found: {args.videos_dir}")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Scanning for video files")
    video_files = get_video_files(args.videos_dir, args.max_folders)
    
    if len(video_files) == 0:
        print(f"No video files found in {args.videos_dir}")
        sys.exit(1)
    
    if args.max_folders:
        print(f"Processing first {args.max_folders} folders")
    
    print(f"Found {len(video_files)} video files")
    
    process_videos_batch(video_files, args.output_dir, args)


if __name__ == "__main__":
    main()
