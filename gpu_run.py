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
from scipy.spatial.distance import cdist

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


def estimate_head_pose_simple(face_landmarks):
    if face_landmarks is None or len(face_landmarks) < 5:
        return 0.0
    
    left_eye = face_landmarks[0]
    right_eye = face_landmarks[1]
    nose = face_landmarks[2]
    
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    
    face_width = abs(right_eye[0] - left_eye[0])
    if face_width < 1:
        return 0.0
    
    nose_offset = (nose[0] - eye_center_x) / face_width
    
    yaw_angle = nose_offset * 90
    
    return abs(yaw_angle)


def compute_quality_score(face_crop, bbox, frame_shape, det_conf, embedding_norm, landmarks=None):
    h, w = face_crop.shape[:2]
    frame_h, frame_w = frame_shape[:2]
    
    if h < 10 or w < 10:
        return 0.0
    
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
    
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness = min(laplacian_var / 2000.0, 1.0)
    
    if laplacian_var < 100:
        sharpness = 0.0
    
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    frame_area = frame_h * frame_w
    relative_size = bbox_area / frame_area
    
    if relative_size < 0.01:
        return 0.0
    
    size_score = min(relative_size / 0.1, 1.0)
    
    x1, y1, x2, y2 = bbox
    border_penalty = 1.0
    margin = 20
    if x1 < margin or y1 < margin or x2 > (frame_w - margin) or y2 > (frame_h - margin):
        border_penalty = 0.0
    
    embedding_score = min(embedding_norm / 20.0, 1.0)
    
    pose_penalty = 1.0
    if landmarks is not None:
        yaw = estimate_head_pose_simple(landmarks)
        if yaw > 30:
            return 0.0
        elif yaw > 20:
            pose_penalty = 0.3
        elif yaw > 10:
            pose_penalty = 0.7
    
    brightness = gray.mean()
    if brightness < 40 or brightness > 220:
        return 0.0
    
    brightness_score = 1.0 - abs(brightness - 127.5) / 127.5
    
    contrast = gray.std()
    if contrast < 20:
        return 0.0
    contrast_score = min(contrast / 60.0, 1.0)
    
    quality = (
        0.35 * sharpness +
        0.25 * size_score +
        0.15 * brightness_score +
        0.10 * contrast_score +
        0.05 * det_conf +
        0.05 * embedding_score +
        0.025 * border_penalty +
        0.025 * pose_penalty
    )
    
    return quality


def simple_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def build_tracklets(face_detections, iou_threshold=0.2, max_frame_gap=30):
    tracklets = []
    active_tracks = []
    
    for frame_idx in sorted(face_detections.keys()):
        current_detections = face_detections[frame_idx]
        
        matched = set()
        
        for track in active_tracks:
            if frame_idx - track['last_frame'] > max_frame_gap:
                tracklets.append(track)
                continue
            
            last_bbox = track['detections'][-1]['bbox']
            
            best_iou = 0
            best_det_idx = -1
            
            for det_idx, det in enumerate(current_detections):
                if det_idx in matched:
                    continue
                iou = simple_iou(last_bbox, det['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx
            
            if best_iou > iou_threshold:
                track['detections'].append(current_detections[best_det_idx])
                track['last_frame'] = frame_idx
                matched.add(best_det_idx)
        
        active_tracks = [t for t in active_tracks if frame_idx - t['last_frame'] <= max_frame_gap]
        
        for det_idx, det in enumerate(current_detections):
            if det_idx not in matched:
                new_track = {
                    'detections': [det],
                    'last_frame': frame_idx,
                    'start_frame': frame_idx
                }
                active_tracks.append(new_track)
    
    tracklets.extend(active_tracks)
    
    return [t for t in tracklets if len(t['detections']) >= 2]


def extract_faces_with_tracking(video_path, sample_rate, min_face_size, app, logger):
    logger.info(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video properties: {total_frames} frames, {fps:.2f} FPS, {width}x{height}")
    
    face_detections = {}
    
    frame_idx = 0
    frames_to_process = total_frames // sample_rate
    
    pbar = tqdm(total=frames_to_process, desc="Extracting faces", unit="frame", leave=False)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_rate == 0:
            faces = app.get(frame)
            
            current_frame_detections = []
            
            for face in faces:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                if (x2 - x1) >= min_face_size and (y2 - y1) >= min_face_size:
                    face_crop = frame[y1:y2, x1:x2]
                    
                    if face_crop.size == 0:
                        continue
                    
                    embedding_norm = np.linalg.norm(face.embedding)
                    
                    landmarks = face.kps if hasattr(face, 'kps') else None
                    
                    quality_score = compute_quality_score(
                        face_crop, 
                        [x1, y1, x2, y2], 
                        frame.shape,
                        float(face.det_score),
                        embedding_norm,
                        landmarks
                    )
                    
                    current_frame_detections.append({
                        'frame_idx': int(frame_idx),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'embedding': face.embedding,
                        'confidence': float(face.det_score),
                        'quality_score': float(quality_score),
                        'embedding_norm': float(embedding_norm)
                    })
            
            if current_frame_detections:
                face_detections[frame_idx] = current_frame_detections
            
            pbar.update(1)
        
        frame_idx += 1
    
    cap.release()
    pbar.close()
    
    logger.info(f"Building tracklets from {len(face_detections)} frames")
    tracklets = build_tracklets(face_detections)
    
    logger.info(f"Built {len(tracklets)} tracklets")
    
    return tracklets, total_frames


def cluster_tracklets(tracklets, logger, total_frames, eps=0.28, min_samples=3):
    logger.info(f"Clustering {len(tracklets)} tracklets")
    
    tracklet_embeddings = []
    for track in tracklets:
        embeddings = np.array([d['embedding'] for d in track['detections']])
        mean_embedding = embeddings.mean(axis=0)
        tracklet_embeddings.append(mean_embedding)
    
    tracklet_embeddings = np.array(tracklet_embeddings)
    
    similarity_matrix = cosine_similarity(tracklet_embeddings)
    distance_matrix = 1 - similarity_matrix
    distance_matrix = np.clip(distance_matrix, 0, None)
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = clustering.fit_predict(distance_matrix)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    cluster_sizes = {}
    for label in labels:
        if label != -1:
            cluster_sizes[label] = cluster_sizes.get(label, 0) + 1
    
    movie_duration_minutes = total_frames / (24 * 60)
    
    if movie_duration_minutes < 60:
        min_tracklets = 2
    elif movie_duration_minutes < 90:
        min_tracklets = 3
    else:
        min_tracklets = 4
    
    main_characters = [label for label, size in cluster_sizes.items() if size >= min_tracklets]
    
    filtered_labels = np.array([-1 if (label not in main_characters) else label for label in labels])
    n_filtered = len(set(filtered_labels)) - (1 if -1 in filtered_labels else 0)
    
    logger.info(f"Clustering: {n_clusters} initial clusters, {n_filtered} characters")
    
    return filtered_labels, tracklet_embeddings


def farthest_point_sampling(embeddings, qualities, k=5):
    n = len(embeddings)
    if n <= k:
        return list(range(n))
    
    selected_indices = [np.argmax(qualities)]
    
    for _ in range(k - 1):
        distances = cdist(embeddings, embeddings[selected_indices], metric='cosine')
        min_distances = distances.min(axis=1)
        
        for idx in selected_indices:
            min_distances[idx] = -np.inf
        
        next_idx = np.argmax(min_distances)
        selected_indices.append(next_idx)
    
    return selected_indices


def has_text_overlay(face_crop):
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, 100, 200)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=5)
    
    if lines is not None and len(lines) > 5:
        horizontal_lines = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 5:
                horizontal_lines += 1
        
        if horizontal_lines > 3:
            return True
    
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    white_pixels = np.sum(binary == 255) / binary.size
    if white_pixels > 0.15:
        return True
    
    return False


def check_color_quality(face_crop):
    hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
    
    v_channel = hsv[:, :, 2]
    avg_brightness = v_channel.mean()
    
    if avg_brightness < 80:
        return False, "too_dark"
    
    s_channel = hsv[:, :, 1]
    avg_saturation = s_channel.mean()
    
    if avg_saturation > 180:
        return False, "oversaturated"
    
    b, g, r = cv2.split(face_crop)
    
    color_balance = np.std([b.mean(), g.mean(), r.mean()])
    if color_balance > 50:
        return False, "color_cast"
    
    return True, "good"


def select_prototype_set(tracklets, labels, tracklet_embeddings, video_path, top_quality_percentile=90, k_prototypes=5):
    character_prototypes = {}
    
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    for label in tqdm(unique_labels, desc="Building prototype sets", leave=False):
        cluster_indices = np.where(labels == label)[0]
        
        all_detections = []
        for idx in cluster_indices:
            all_detections.extend(tracklets[idx]['detections'])
        
        if not all_detections:
            continue
        
        cluster_centroid = tracklet_embeddings[cluster_indices].mean(axis=0)
        
        candidate_faces = []
        
        for det in all_detections:
            frame_idx = det['frame_idx']
            bbox = det['bbox']
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret or frame is None:
                continue
            
            x1, y1, x2, y2 = bbox
            
            face_width = x2 - x1
            face_height = y2 - y1
            
            if face_width < 60 or face_height < 60:
                continue
            
            padding_h = int(face_height * 0.25)
            padding_w = int(face_width * 0.25)
            
            x1 = max(0, x1 - padding_w)
            y1 = max(0, y1 - padding_h)
            x2 = min(width, x2 + padding_w)
            y2 = min(height, y2 + padding_h)
            
            margin = 15
            if x1 < margin or y1 < margin or x2 > (width - margin) or y2 > (height - margin):
                continue
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                continue
            
            crop_h, crop_w = face_crop.shape[:2]
            
            if crop_h < 60 or crop_w < 60:
                continue
            
            if has_text_overlay(face_crop):
                continue
            
            color_ok, color_reason = check_color_quality(face_crop)
            if not color_ok and color_reason == "too_dark":
                continue
            
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 100:
                continue
            
            brightness = gray.mean()
            if brightness < 50 or brightness > 200:
                continue
            
            contrast = gray.std()
            if contrast < 20:
                continue
            
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_mag = np.sqrt(sobelx**2 + sobely**2)
            edge_strength = edge_mag.mean()
            
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_norm = hist.ravel() / hist.sum()
            entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-7))
            
            hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1].mean()
            
            embedding_dist = np.linalg.norm(det['embedding'] - cluster_centroid)
            embedding_score = 1.0 / (1.0 + embedding_dist)
            
            visual_quality = (
                0.35 * min(laplacian_var / 1000.0, 1.0) +
                0.25 * min(edge_strength / 50.0, 1.0) +
                0.15 * (1.0 - abs(brightness - 127.5) / 127.5) +
                0.15 * min(contrast / 60.0, 1.0) +
                0.10 * min(entropy / 8.0, 1.0)
            )
            
            final_score = 0.60 * visual_quality + 0.40 * embedding_score
            
            candidate_faces.append({
                'detection': det,
                'crop': face_crop,
                'visual_quality': visual_quality,
                'embedding_score': embedding_score,
                'final_score': final_score,
                'frame_idx': frame_idx,
                'bbox': [x1, y1, x2, y2],
                'sharpness': laplacian_var
            })
        
        if not candidate_faces:
            continue
        
        candidate_faces.sort(key=lambda x: x['final_score'], reverse=True)
        
        top_candidates = candidate_faces[:min(k_prototypes * 10, len(candidate_faces))]
        
        selected_prototypes = []
        selected_embeddings = []
        
        if top_candidates:
            best = top_candidates[0]
            selected_prototypes.append(best)
            selected_embeddings.append(best['detection']['embedding'])
        
        for candidate in top_candidates[1:]:
            if len(selected_prototypes) >= k_prototypes:
                break
            
            min_dist = float('inf')
            for sel_emb in selected_embeddings:
                dist = np.linalg.norm(candidate['detection']['embedding'] - sel_emb)
                min_dist = min(min_dist, dist)
            
            if min_dist > 0.3:
                selected_prototypes.append(candidate)
                selected_embeddings.append(candidate['detection']['embedding'])
        
        if selected_prototypes:
            prototypes_data = []
            for proto in selected_prototypes:
                prototypes_data.append({
                    'frame_idx': proto['frame_idx'],
                    'bbox': proto['bbox'],
                    'embedding': proto['detection']['embedding'],
                    'confidence': proto['detection']['confidence'],
                    'quality_score': proto['visual_quality']
                })
            
            character_prototypes[f"character_{label}"] = {
                'prototypes': prototypes_data,
                'mean_embedding': cluster_centroid,
                'num_tracklets': len(cluster_indices),
                'total_detections': len(all_detections)
            }
    
    cap.release()
    return character_prototypes


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


def extract_prototype_images(video_path, character_prototypes, output_dir, logger):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info("Extracting prototype images")
    
    prototype_images = {}
    
    for char_id, data in tqdm(character_prototypes.items(), desc="Extracting prototypes", leave=False):
        char_images = []
        
        for proto_idx, proto in enumerate(data['prototypes']):
            frame_idx = proto['frame_idx']
            bbox = proto['bbox']
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            x1, y1, x2, y2 = bbox
            
            face_width = x2 - x1
            face_height = y2 - y1
            
            padding_h = int(face_height * 0.5)
            padding_w = int(face_width * 0.5)
            
            x1 = max(0, x1 - padding_w)
            y1 = max(0, y1 - padding_h)
            x2 = min(width, x2 + padding_w)
            y2 = min(height, y2 + padding_h)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                continue
            
            crop_h, crop_w = face_crop.shape[:2]
            
            if crop_h < 200 or crop_w < 200:
                target_size = 256
            else:
                target_size = min(512, max(crop_h, crop_w))
            
            if crop_h > target_size or crop_w > target_size:
                enhanced = enhance_image(face_crop)
                aspect_ratio = crop_w / crop_h
                if aspect_ratio > 1:
                    new_w = target_size
                    new_h = int(target_size / aspect_ratio)
                else:
                    new_h = target_size
                    new_w = int(target_size * aspect_ratio)
                enhanced_resized = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            else:
                enhanced_resized = enhance_image(face_crop)
            
            face_rgb = cv2.cvtColor(enhanced_resized, cv2.COLOR_BGR2RGB)
            
            ref_path = f"{output_dir}/{char_id}_proto_{proto_idx}.jpg"
            Image.fromarray(face_rgb).save(ref_path, quality=95)
            char_images.append(face_rgb)
        
        if char_images:
            prototype_images[char_id] = char_images[0]
    
    cap.release()
    logger.info(f"Extracted prototypes for {len(prototype_images)} characters")
    return prototype_images


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


def match_faces_in_frame(frame, character_prototypes, similarity_threshold, app):
    faces = app.get(frame)
    
    matched_faces = []
    
    for face in faces:
        bbox = face.bbox.astype(int)
        embedding = face.embedding
        
        best_match = None
        best_similarity = -1
        all_similarities = {}
        
        for char_id, char_data in character_prototypes.items():
            max_sim = -1
            
            for proto in char_data['prototypes']:
                proto_embedding = proto['embedding']
                similarity = cosine_similarity([embedding], [proto_embedding])[0][0]
                max_sim = max(max_sim, similarity)
            
            all_similarities[char_id] = max_sim
            
            if max_sim > best_similarity:
                best_similarity = max_sim
                if max_sim > similarity_threshold:
                    best_match = char_id
        
        matched_faces.append({
            'bbox': bbox.tolist(),
            'character_id': best_match,
            'confidence': float(face.det_score),
            'similarity': float(best_similarity),
            'all_similarities': {k: float(v) for k, v in all_similarities.items()}
        })
    
    return matched_faces


def create_visualization(frame, matches, prototype_images, frame_idx):
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
        if char_id in prototype_images:
            axes[idx].imshow(prototype_images[char_id])
            axes[idx].set_title(char_id.replace('character_', 'Character '), fontsize=12)
            axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def generate_summary_report(frame_matches, character_prototypes):
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

CHARACTER DATABASE:
- Total unique characters: {len(character_prototypes)}

CHARACTER APPEARANCES IN TEST FRAMES:
"""
    
    for char_id in sorted(char_appearances.keys()):
        count = char_appearances[char_id]
        percentage = count / total_frames * 100
        num_prototypes = len(character_prototypes[char_id]['prototypes'])
        report += f"- {char_id}: {count} frames ({percentage:.1f}%) [{num_prototypes} prototypes]\n"
    
    return report


@ray.remote(num_gpus=0.25, num_cpus=1)
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
        
        import torch
        if torch.cuda.is_available() and ray.get_gpu_ids():
            gpu_id = int(ray.get_gpu_ids()[0])
        else:
            gpu_id = 0
        
        app = FaceAnalysis(
            name='buffalo_l',
            providers=providers
        )
        app.prepare(ctx_id=gpu_id, det_size=(640, 640))
        
        logger.info(f"Model: buffalo_l on GPU {gpu_id}")
        
        tracklets, total_frames = extract_faces_with_tracking(
            video_path, args['sample_rate'], args['min_face_size'], app, logger
        )
        
        if len(tracklets) == 0:
            logger.warning("No tracklets found in video")
            return False
        
        labels, tracklet_embeddings = cluster_tracklets(tracklets, logger, total_frames)
        
        character_prototypes = select_prototype_set(
            tracklets, labels, tracklet_embeddings, video_path,
            top_quality_percentile=args['quality_percentile'],
            k_prototypes=args['k_prototypes']
        )
        
        logger.info(f"Built prototype sets for {len(character_prototypes)} characters")
        
        prototype_images = extract_prototype_images(video_path, character_prototypes, output_dir, logger)
        
        test_frames = generate_test_frames_from_video(video_path, args['num_test_frames'], output_dir, logger)
        
        viz_dir = f"{output_dir}/visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        logger.info("Matching faces and creating visualizations")
        frame_matches = []
        
        for frame_idx, frame_path, frame in tqdm(test_frames, desc="Processing test frames", leave=False):
            matches = match_faces_in_frame(frame, character_prototypes, args['similarity_threshold'], app)
            
            frame_matches.append({
                'frame_idx': int(frame_idx),
                'frame_path': frame_path,
                'matches': matches
            })
            
            fig = create_visualization(frame, matches, prototype_images, frame_idx)
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
        
        with open(f"{output_dir}/character_prototypes.pkl", 'wb') as f:
            pickle.dump(character_prototypes, f)
        
        summary_report = generate_summary_report(frame_matches, character_prototypes)
        
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
    print(f"Model: buffalo_l + tracklet-based prototypes")
    print(f"Prototypes per character: {args.k_prototypes}")
    print(f"Quality filtering: top {args.quality_percentile}%")
    print(f"GPUs: 2")
    print(f"Max concurrent tasks: 8")
    
    ray.init(num_gpus=2, num_cpus=8)
    
    args_dict = {
        'num_test_frames': args.num_test_frames,
        'sample_rate': args.sample_rate,
        'min_face_size': args.min_face_size,
        'similarity_threshold': args.similarity_threshold,
        'k_prototypes': args.k_prototypes,
        'quality_percentile': args.quality_percentile
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
    parser = argparse.ArgumentParser(description='Research-Grade Movie Character Recognition Pipeline')
    
    parser.add_argument('--videos-dir', type=str, default='/mnt/data/data_hub/movies_dataset/videos')
    parser.add_argument('--output-dir', type=str, default='/mnt/data/data_hub/movies_dataset/face_extraction')
    parser.add_argument('--num-test-frames', type=int, default=150)
    parser.add_argument('--sample-rate', type=int, default=30)
    parser.add_argument('--min-face-size', type=int, default=50)
    parser.add_argument('--similarity-threshold', type=float, default=0.20)
    parser.add_argument('--max-folders', type=int, default=None)
    parser.add_argument('--k-prototypes', type=int, default=5)
    parser.add_argument('--quality-percentile', type=int, default=90)
    
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
