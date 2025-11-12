import argparse
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
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


def detect_anime_faces(frame):
    try:
        from anime_face_detector import create_detector
        if not hasattr(detect_anime_faces, 'detector'):
            detect_anime_faces.detector = create_detector('yolov3')
        
        preds = detect_anime_faces.detector(frame)
        
        anime_faces = []
        for pred in preds:
            bbox = pred['bbox']
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            face_region = frame[y1:y2, x1:x2]
            if face_region.size == 0:
                continue
            
            face_region_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            sift = cv2.SIFT_create(nfeatures=128)
            keypoints, descriptors = sift.detectAndCompute(face_region_gray, None)
            
            if descriptors is not None and len(descriptors) > 0:
                feature = descriptors.mean(axis=0)
                feature = feature / (np.linalg.norm(feature) + 1e-8)
            else:
                face_resized = cv2.resize(face_region, (112, 112))
                feature = face_resized.flatten()
                feature = feature / (np.linalg.norm(feature) + 1e-8)
            
            anime_faces.append({
                'bbox': [x1, y1, x2, y2],
                'embedding': feature,
                'confidence': pred.get('score', 0.5)
            })
        
        return anime_faces
    except Exception as e:
        return []


def detect_content_type(video_path):
    filename = os.path.basename(video_path).lower()
    
    anime_keywords = [
        'anime', 'dragon ball', 'naruto', 'one piece', 'pokemon', 'digimon',
        'evangelion', 'akira', 'ghibli', 'miyazaki', 'spirited away', 'totoro',
        'cowboy bebop', 'ghost in the shell', 'sailor moon', 'bleach',
        'attack on titan', 'demon slayer', 'jujutsu kaisen', 'fate stay',
        'sword art', 'gundam', 'macross', 'lupin', 'detective conan',
        'studio ghibli', 'arrietty', 'ponyo', 'howl', 'mononoke', 'kiki'
    ]
    
    cartoon_keywords = [
        'charlie brown', 'snoopy', 'peanuts', 'tom and jerry', 'bugs bunny',
        'looney', 'scooby', 'garfield', 'simpsons', 'family guy', 'futurama',
        'south park', 'spongebob', 'adventure time', 'regular show',
        'gravity falls', 'steven universe', 'avatar', 'teen titans',
        'lego', 'mickey mouse', 'disney', 'pixar', 'dreamworks',
        'minions', 'despicable', 'shrek', 'kung fu panda', 'madagascar',
        'ice age', 'rio', 'trolls', 'boss baby', 'croods',
        'puss in boots', 'barbie', 'my little pony', 'care bears'
    ]
    
    animation_keywords = [
        'animated', 'animation', 'cartoon', 'claymation', 'stop motion',
        'cgi', 'computer generated', '3d animated', '2d animated'
    ]
    
    for keyword in anime_keywords:
        if keyword in filename:
            return 'anime'
    
    for keyword in cartoon_keywords:
        if keyword in filename:
            return 'cartoon'
    
    for keyword in animation_keywords:
        if keyword in filename:
            return 'animation'
    
    return 'live_action'


def adaptive_face_detection(frame, app, min_face_size, content_type='live_action'):
    width, height = frame.shape[1], frame.shape[0]
    
    faces_real = app.get(frame)
    
    valid_faces_real = []
    for face in faces_real:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        
        if (x2 - x1) >= min_face_size and (y2 - y1) >= min_face_size:
            valid_faces_real.append({
                'bbox': [x1, y1, x2, y2],
                'embedding': face.embedding,
                'confidence': float(face.det_score),
                'face_type': 'real'
            })
    
    if len(valid_faces_real) > 0:
        return valid_faces_real
    
    anime_faces = detect_anime_faces(frame)
    
    valid_faces_anime = []
    for face in anime_faces:
        x1, y1, x2, y2 = face['bbox']
        if (x2 - x1) >= min_face_size and (y2 - y1) >= min_face_size:
            valid_faces_anime.append({
                'bbox': face['bbox'],
                'embedding': face['embedding'],
                'confidence': face['confidence'],
                'face_type': 'anime'
            })
    
    return valid_faces_anime


def extract_faces_from_video(video_path, sample_rate, min_face_size, app, logger):
    logger.info(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video info - Frames: {total_frames}, FPS: {fps:.2f}, Resolution: {width}x{height}")
    
    all_faces = []
    frame_indices = []
    face_boxes = []
    
    frame_idx = 0
    frames_to_process = total_frames // sample_rate
    
    pbar = tqdm(total=frames_to_process, desc="Extracting faces", leave=False)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_rate == 0:
            detected_faces = adaptive_face_detection(frame, app, min_face_size)
            
            for face in detected_faces:
                x1, y1, x2, y2 = face['bbox']
                face_crop = frame[y1:y2, x1:x2]
                
                if face_crop.size == 0:
                    continue
                
                quality_score = calculate_multi_scale_quality(face_crop)
                
                all_faces.append(face['embedding'])
                frame_indices.append(frame_idx)
                face_boxes.append({
                    'frame_idx': int(frame_idx),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(face['confidence']),
                    'quality_score': float(quality_score),
                    'face_type': face['face_type']
                })
            
            pbar.update(1)
        
        frame_idx += 1
    
    cap.release()
    pbar.close()
    
    logger.info(f"Extraction complete - Processed: {frame_idx} frames, Extracted: {len(all_faces)} faces")
    
    return np.array(all_faces), frame_indices, face_boxes


def hierarchical_clustering(embeddings, face_boxes, eps_range=[0.3, 0.4, 0.5], min_samples=3):
    best_labels = None
    best_n_clusters = 0
    best_eps = eps_range[0]
    
    similarity_matrix = cosine_similarity(embeddings)
    distance_matrix = 1 - similarity_matrix
    distance_matrix = np.clip(distance_matrix, 0, None)
    
    for eps in eps_range:
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        labels = clustering.fit_predict(distance_matrix)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters > best_n_clusters:
            best_n_clusters = n_clusters
            best_labels = labels
            best_eps = eps
    
    return best_labels, best_eps


def cluster_faces(embeddings, face_boxes, logger):
    logger.info(f"Clustering {len(embeddings)} faces with adaptive parameters...")
    
    labels, best_eps = hierarchical_clustering(embeddings, face_boxes)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    logger.info(f"Clustering complete - Characters: {n_clusters}, Noise: {n_noise}, Best eps: {best_eps}")
    
    return labels


def select_best_reference_faces(embeddings, labels, face_boxes, quality_weight=0.6):
    reference_faces = {}
    
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    
    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]
        cluster_boxes = [face_boxes[i] for i in cluster_indices]
        
        scores = []
        for box in cluster_boxes:
            combined_score = (box['confidence'] * (1 - quality_weight)) + (box['quality_score'] * quality_weight)
            scores.append(combined_score)
        
        best_idx = cluster_indices[np.argmax(scores)]
        
        reference_faces[f"character_{label}"] = {
            'embedding': embeddings[best_idx],
            'frame_idx': face_boxes[best_idx]['frame_idx'],
            'bbox': face_boxes[best_idx]['bbox'],
            'confidence': face_boxes[best_idx]['confidence'],
            'quality_score': face_boxes[best_idx]['quality_score'],
            'face_type': face_boxes[best_idx]['face_type'],
            'cluster_size': len(cluster_indices)
        }
    
    return reference_faces


def extract_reference_images(video_path, reference_faces, output_dir, logger):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ref_images = {}
    
    logger.info("Extracting reference face images...")
    
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
                
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            
            ref_path = f"{output_dir}/{char_id}_reference.jpg"
            Image.fromarray(face_crop).save(ref_path, quality=95)
            ref_images[char_id] = face_crop
    
    cap.release()
    logger.info(f"Extracted {len(ref_images)} reference images")
    return ref_images


def generate_test_frames_from_video(video_path, num_frames, output_dir, logger):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    test_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    test_frames_dir = f"{output_dir}/test_frames"
    os.makedirs(test_frames_dir, exist_ok=True)
    
    logger.info(f"Generating {num_frames} test frames...")
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
    detected_faces = adaptive_face_detection(frame, app, 30)
    
    matched_faces = []
    
    for face in detected_faces:
        bbox = face['bbox']
        embedding = face['embedding']
        face_type = face['face_type']
        
        best_match = None
        best_similarity = -1
        
        for char_id, ref_data in reference_faces.items():
            if ref_data.get('face_type') != face_type:
                continue
            
            ref_embedding = ref_data['embedding']
            similarity = cosine_similarity([embedding], [ref_embedding])[0][0]
            
            if similarity > best_similarity and similarity > similarity_threshold:
                best_similarity = similarity
                best_match = char_id
        
        matched_faces.append({
            'bbox': bbox,
            'character_id': best_match,
            'confidence': float(face['confidence']),
            'similarity': float(best_similarity) if best_match else 0.0
        })
    
    return matched_faces


def create_visualization(frame, matches, reference_images, frame_idx):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    frame_annotated = frame_rgb.copy()
    pil_frame = Image.fromarray(frame_annotated)
    draw = ImageDraw.Draw(pil_frame)
    
    detected_chars = []
    
    for match in matches:
        x1, y1, x2, y2 = match['bbox']
        char_id = match['character_id']
        
        if char_id:
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
            detected_chars.append(char_id)
    
    frame_annotated = np.array(pil_frame)
    
    if len(matches) == 0:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(frame_annotated)
        ax.set_title(f"Frame {frame_idx}: NO FACES FOUND", fontsize=16, color='red', weight='bold')
        ax.axis('off')
        ax.text(0.5, 0.5, 'NO FACES FOUND', 
                transform=ax.transAxes, fontsize=30, color='red',
                ha='center', va='center', weight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        return fig
    
    num_detected = len(detected_chars)
    
    if num_detected == 0:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(frame_annotated)
        ax.set_title(f"Frame {frame_idx}: Faces detected but not matched", 
                     fontsize=14, weight='bold')
        ax.axis('off')
        return fig
    
    cols = num_detected + 1
    fig_width = 4 * cols
    fig, axes = plt.subplots(1, cols, figsize=(fig_width, 6))
    
    if cols == 2:
        axes = [axes[0], axes[1]]
    
    axes[0].imshow(frame_annotated)
    axes[0].set_title("Input Frame", fontsize=14, weight='bold')
    axes[0].text(0.5, -0.05, 'im', transform=axes[0].transAxes, 
                 fontsize=10, ha='center')
    axes[0].axis('off')
    
    for idx, char_id in enumerate(detected_chars, 1):
        if char_id in reference_images:
            axes[idx].imshow(reference_images[char_id])
            axes[idx].set_title(char_id.replace('character_', 'Character '), 
                                fontsize=12, weight='bold')
            axes[idx].text(0.5, -0.05, 'ref', transform=axes[idx].transAxes,
                           fontsize=10, ha='center')
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
            name='buffalo_l',
            providers=providers
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        logger.info("Model loaded successfully")
        
        embeddings, frame_indices, face_boxes = extract_faces_from_video(
            video_path, args['sample_rate'], args['min_face_size'], app, logger
        )
        
        if len(embeddings) == 0:
            logger.warning("No faces found in video")
            return False
        
        labels = cluster_faces(embeddings, face_boxes, logger)
        
        reference_faces = select_best_reference_faces(embeddings, labels, face_boxes, args['quality_weight'])
        logger.info(f"Selected {len(reference_faces)} reference faces")
        
        reference_images = extract_reference_images(video_path, reference_faces, output_dir, logger)
        
        test_frames = generate_test_frames_from_video(video_path, args['num_test_frames'], output_dir, logger)
        
        viz_dir = f"{output_dir}/visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        logger.info("Matching faces and creating visualizations...")
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
    
    print(f"\nStarting batch processing with Ray")
    print(f"Total videos: {total_videos}")
    print(f"GPUs: 2 (0.0625 GPU per task)")
    print(f"Concurrent pipelines: 32")
    print(f"Quality weight: {args.quality_weight}")
    print(f"Anime detection: Adaptive (auto-enabled when no real faces found)")
    
    ray.init(num_gpus=2, num_cpus=32)
    
    args_dict = {
        'num_test_frames': args.num_test_frames,
        'sample_rate': args.sample_rate,
        'min_face_size': args.min_face_size,
        'similarity_threshold': args.similarity_threshold,
        'quality_weight': args.quality_weight
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
    parser = argparse.ArgumentParser(description='Adaptive Multi-Modal Face Recognition Pipeline')
    
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
    parser.add_argument('--quality-weight', type=float, default=0.6,
                        help='Weight for image quality in reference selection (0-1)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.videos_dir):
        print(f"Error: Videos directory not found: {args.videos_dir}")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Scanning for video files...")
    video_files = get_video_files(args.videos_dir, args.max_folders)
    
    if len(video_files) == 0:
        print(f"No video files found in {args.videos_dir}")
        sys.exit(1)
    
    if args.max_folders:
        print(f"Processing first {args.max_folders} folders for testing")
    
    print(f"Found {len(video_files)} video files")
    
    process_videos_batch(video_files, args.output_dir, args)


if __name__ == "__main__":
    main()
