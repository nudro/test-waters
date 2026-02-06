"""
Vessel Detection Server - FastAPI Backend
Provides WebSocket streaming for video frames and detection data.
"""

import cv2
import numpy as np
import json
import math
import asyncio
import base64
from pathlib import Path
from typing import Dict, Set, List, Optional, Tuple
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import torch
import glob

app = FastAPI(title="Vessel Detection Server")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
video_cap: Optional[cv2.VideoCapture] = None
yolo_model: Optional[YOLO] = None
active_connections: Set[WebSocket] = set()
processing_state = {
    "video_path": None,
    "current_frame": None,
    "frame_count": 0,
    "fps": 30,
    "width": 0,
    "height": 0,
    "is_paused": True,
    "is_tracking": False,
    "selected_track_ids": set(),
    "vessel_tracks": {},  # {track_id: {centroids, velocities, speeds, accelerations, directions, bboxes, frames}}
    "active_track_ids": set(),
    "next_track_id": 0,
    "track_last_seen": {},
}

# Motion calculation parameters
SMOOTHING_FACTOR = 0.7
MAX_VELOCITY_CHANGE_RATIO = 2.0
MAX_PIXEL_CHANGE = 100
IOU_THRESHOLD = 0.3
MAX_FRAMES_WITHOUT_DETECTION = 10
CONF_THRESHOLD = 0.25


# ============================================================================
# Motion Calculation Functions (from archive)
# ============================================================================

def compute_bbox_centroid(bbox):
    """Compute centroid from bounding box [x1, y1, x2, y2]."""
    if bbox is None or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = bbox[:4]
    centroid_x = (x1 + x2) / 2.0
    centroid_y = (y1 + y2) / 2.0
    return (centroid_x, centroid_y)


def compute_bbox_area(bbox):
    """Compute area of bounding box in pixels."""
    if bbox is None or len(bbox) < 4:
        return 0
    x1, y1, x2, y2 = bbox[:4]
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return int(width * height)


def detect_vessels_with_yolo(frame, yolo_model, conf_threshold=0.25):
    """Use YOLO to detect vessels in a frame and return bounding boxes."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = yolo_model(frame, conf=conf_threshold, verbose=False, device=device)
    bboxes = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            bboxes.append([int(x1), int(y1), int(x2), int(y2), confidence])
    
    return bboxes


def compute_velocity(prev_centroid, curr_centroid, dt):
    """Compute velocity vector from centroid positions."""
    if prev_centroid is None or curr_centroid is None:
        return None
    vx = (curr_centroid[0] - prev_centroid[0]) / dt
    vy = (curr_centroid[1] - prev_centroid[1]) / dt
    return (vx, vy)


def compute_speed(velocity):
    """Compute speed magnitude from velocity vector."""
    if velocity is None:
        return None
    return math.sqrt(velocity[0]**2 + velocity[1]**2)


def compute_acceleration(prev_velocity, curr_velocity, dt):
    """Compute acceleration vector from velocity vectors."""
    if prev_velocity is None or curr_velocity is None:
        return None
    ax = (curr_velocity[0] - prev_velocity[0]) / dt
    ay = (curr_velocity[1] - curr_velocity[1]) / dt
    return (ax, ay)


def compute_direction(velocity):
    """Compute direction angle in degrees from velocity vector."""
    if velocity is None:
        return None
    angle_rad = math.atan2(velocity[1], velocity[0])
    angle_deg = math.degrees(angle_rad)
    return angle_deg


def smooth_velocity(curr_velocity, prev_velocities, smoothing_factor=0.7, max_change_ratio=2.0):
    """Smooth velocity using exponential moving average and outlier detection."""
    if curr_velocity is None:
        return None
    if not prev_velocities:
        return curr_velocity
    
    prev_velocity = prev_velocities[-1]
    curr_speed = math.sqrt(curr_velocity[0]**2 + curr_velocity[1]**2)
    prev_speed = math.sqrt(prev_velocity[0]**2 + prev_velocity[1]**2)
    
    if prev_speed > 0:
        speed_change_ratio = curr_speed / prev_speed
        if speed_change_ratio > max_change_ratio or speed_change_ratio < (1.0 / max_change_ratio):
            smoothed_vx = prev_velocity[0] * (1 - smoothing_factor) + curr_velocity[0] * smoothing_factor
            smoothed_vy = prev_velocity[1] * (1 - smoothing_factor) + curr_velocity[1] * smoothing_factor
            return (smoothed_vx, smoothed_vy)
    
    smoothed_vx = prev_velocity[0] * (1 - smoothing_factor) + curr_velocity[0] * smoothing_factor
    smoothed_vy = prev_velocity[1] * (1 - smoothing_factor) + curr_velocity[1] * smoothing_factor
    return (smoothed_vx, smoothed_vy)


def validate_centroid_change(prev_centroid, curr_centroid, max_pixel_change=100):
    """Validate that centroid change is reasonable (outlier detection)."""
    if prev_centroid is None or curr_centroid is None:
        return True
    dx = curr_centroid[0] - prev_centroid[0]
    dy = curr_centroid[1] - prev_centroid[1]
    distance = math.sqrt(dx**2 + dy**2)
    return distance <= max_pixel_change


def compute_iou(bbox1, bbox2):
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def match_detections_to_tracks(new_detections, existing_tracks, iou_threshold=0.3):
    """Match new YOLO detections to existing tracks using IoU."""
    matches = {}  # {track_id: detection_index}
    unmatched_detections = list(range(len(new_detections)))
    
    if not existing_tracks or not new_detections:
        return matches, unmatched_detections
    
    # Compute IoU matrix
    iou_matrix = np.zeros((len(existing_tracks), len(new_detections)))
    
    for track_idx, (track_id, track_data) in enumerate(existing_tracks.items()):
        if not track_data.get("bboxes"):
            continue
        prev_bbox = track_data["bboxes"][-1]
        
        for det_idx, det_bbox in enumerate(new_detections):
            iou = compute_iou(prev_bbox[:4], det_bbox[:4])
            iou_matrix[track_idx, det_idx] = iou
    
    # Greedy matching
    used_tracks = set()
    used_detections = set()
    matches_list = []
    
    for track_idx in range(len(existing_tracks)):
        for det_idx in range(len(new_detections)):
            if iou_matrix[track_idx, det_idx] >= iou_threshold:
                matches_list.append((track_idx, det_idx, iou_matrix[track_idx, det_idx]))
    
    matches_list.sort(key=lambda x: x[2], reverse=True)
    
    for track_idx, det_idx, iou in matches_list:
        if track_idx not in used_tracks and det_idx not in used_detections:
            track_id = list(existing_tracks.keys())[track_idx]
            matches[track_id] = det_idx
            used_tracks.add(track_idx)
            used_detections.add(det_idx)
            if det_idx in unmatched_detections:
                unmatched_detections.remove(det_idx)
    
    return matches, unmatched_detections


def resolve_video_path(video_path, search_media=True):
    """Resolve video path - only searches in project media directory."""
    video_path_obj = Path(video_path)
    
    # If absolute path exists, return it
    if video_path_obj.is_absolute() and video_path_obj.exists():
        return video_path_obj
    
    # If relative path exists, return it
    if not video_path_obj.is_absolute() and video_path_obj.exists():
        return video_path_obj.resolve()
    
    if not search_media:
        raise ValueError(f"Video file not found: {video_path}")
    
    # Only search in project media directory
    script_dir = Path(__file__).parent
    project_media = script_dir / "media"
    
    if not project_media.exists():
        raise ValueError(f"Media directory not found: {project_media}")
    
    # Get filename
    video_filename = video_path_obj.name if video_path_obj.name else str(video_path_obj)
    
    # Try exact match first
    video_file = project_media / video_filename
    if video_file.exists() and video_file.is_file():
        return video_file
    
    # Try with .mp4 extension if not present
    if not video_filename.lower().endswith('.mp4'):
        video_file = project_media / f"{video_filename}.mp4"
        if video_file.exists() and video_file.is_file():
            return video_file
    
    # Try case-insensitive match
    for file in project_media.glob("*.mp4"):
        if file.name.lower() == video_filename.lower() or file.name.lower() == f"{video_filename}.mp4".lower():
            return file
    
    raise ValueError(f"Video file not found in {project_media}: {video_path}")


def frame_to_base64(frame):
    """Convert OpenCV frame to base64 JPEG string."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    return frame_base64


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    return {"message": "Vessel Detection Server", "status": "running"}


@app.get("/api/videos")
async def list_videos():
    """List available videos in project media directory only."""
    videos = []
    script_dir = Path(__file__).parent
    project_media = script_dir / "media"
    
    if not project_media.exists():
        return {"videos": []}
    
    # Only search in project media directory, only .mp4 files
    for video_file in project_media.glob("*.mp4"):
        if video_file.is_file():
            # Validate video can be opened
            try:
                test_cap = cv2.VideoCapture(str(video_file))
                if test_cap.isOpened():
                    # Try to read first frame to ensure video is valid
                    ret, _ = test_cap.read()
                    test_cap.release()
                    if ret:
                        videos.append({
                            "name": video_file.name,
                            "path": str(video_file),
                            "size": video_file.stat().st_size
                        })
            except Exception as e:
                print(f"Warning: Skipping {video_file.name} - {e}")
                continue
    
    return {"videos": videos}


@app.post("/api/load-video")
async def load_video(data: dict):
    """Load a video file and prepare for processing."""
    global video_cap, processing_state
    
    video_path = data.get("video_path")
    if not video_path:
        raise HTTPException(status_code=400, detail="video_path is required")
    
    try:
        video_path = resolve_video_path(video_path, search_media=True)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    # Close existing video if open
    if video_cap is not None:
        video_cap.release()
    
    # Open new video with validation
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail=f"Could not open video: {video_path}. Video file may be corrupted or in an unsupported format.")
    
    # Validate video by reading first frame
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        cap.release()
        raise HTTPException(status_code=500, detail=f"Video file appears to be corrupted or empty: {video_path}. Cannot read frames.")
    
    # Reset to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    video_cap = cap
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Validate video properties
    if width == 0 or height == 0:
        cap.release()
        raise HTTPException(status_code=500, detail=f"Invalid video dimensions: {width}x{height}. Video may be corrupted.")
    
    processing_state.update({
        "video_path": str(video_path),
        "frame_count": 0,
        "fps": fps,
        "width": width,
        "height": height,
        "is_paused": False,  # New video = fresh state, not paused
        "is_tracking": False,
        "auto_playing": True,  # Auto-play a few frames to get detections
        "auto_play_frames": 10,  # Number of frames to auto-play
        "auto_play_count": 0,  # Counter for auto-play
        "selected_track_ids": set(),
        "vessel_tracks": {},
        "active_track_ids": set(),
        "next_track_id": 0,
        "track_last_seen": {},
    })
    
    # Read first frame for display
    ret, frame = cap.read()
    if ret and frame is not None:
        processing_state["current_frame"] = frame
        processing_state["frame_count"] = 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
    else:
        cap.release()
        raise HTTPException(status_code=500, detail=f"Could not read first frame from video: {video_path}")
    
    return {
        "success": True,
        "video_path": str(video_path),
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames
    }


@app.post("/api/tag")
async def tag_vessels(data: dict):
    """Tag selected track IDs for tracking."""
    global processing_state
    
    track_ids = data.get("track_ids", [])
    if not track_ids:
        raise HTTPException(status_code=400, detail="track_ids is required")
    
    processing_state["selected_track_ids"] = set(track_ids)
    print(f"✓ Tagged {len(track_ids)} vessel(s): {sorted(track_ids)}")
    
    return {
        "success": True,
        "selected_track_ids": list(processing_state["selected_track_ids"])
    }


@app.post("/api/track")
async def start_tracking():
    """Start tracking selected vessels."""
    global processing_state
    
    selected_ids = processing_state.get("selected_track_ids", set())
    print(f"DEBUG: Checking selected_track_ids: {selected_ids}, type: {type(selected_ids)}, empty: {not selected_ids}")
    
    if not selected_ids or len(selected_ids) == 0:
        raise HTTPException(status_code=400, detail="No vessels selected. Please tag vessels first.")
    
    processing_state["is_tracking"] = True
    processing_state["is_paused"] = False
    print(f"✓ Started tracking {len(selected_ids)} vessel(s): {sorted(selected_ids)}")
    
    return {
        "success": True,
        "message": "Tracking started",
        "selected_track_ids": list(selected_ids)
    }


@app.post("/api/pause")
async def pause_tracking():
    """Pause video tracking."""
    global processing_state
    processing_state["is_paused"] = True
    processing_state["is_tracking"] = False
    
    return {
        "success": True,
        "message": "Tracking paused"
    }


@app.post("/api/all-detections")
async def start_all_detections():
    """Start tracking all detected vessels automatically."""
    global processing_state
    
    # Set mode to track all vessels
    processing_state["is_tracking"] = True
    processing_state["is_paused"] = False
    # Don't filter by selected_track_ids - track all active vessels
    print("✓ Started tracking all detections")
    
    return {
        "success": True,
        "message": "Tracking all detections"
    }


# ============================================================================
# WebSocket Connection
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming frames and detection data."""
    global processing_state, yolo_model
    
    await websocket.accept()
    active_connections.add(websocket)
    
    try:
        # Load YOLO model if not loaded
        if yolo_model is None:
            model_path = Path("models/yolov8n_vessels.pt")
            if not model_path.exists():
                await websocket.send_json({
                    "type": "error",
                    "message": f"YOLO model not found: {model_path}"
                })
                return
            yolo_model = YOLO(str(model_path))
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            yolo_model.to(device)
            await websocket.send_json({
                "type": "model_loaded",
                "device": device
            })
        
        # Send first frame if available
        if processing_state["current_frame"] is not None and processing_state["frame_count"] == 1:
            frame = processing_state["current_frame"]
            frame_base64 = frame_to_base64(frame)
            
            # Run YOLO on first frame
            detections = detect_vessels_with_yolo(frame, yolo_model, CONF_THRESHOLD)
            
            # Assign track IDs to detections
            detections_with_ids = []
            for i, det in enumerate(detections):
                track_id = processing_state["next_track_id"]
                processing_state["next_track_id"] += 1
                bbox = det[:4]
                centroid = compute_bbox_centroid(bbox)
                detections_with_ids.append({
                    "track_id": track_id,
                    "bbox": bbox,
                    "confidence": det[4],
                    "speed": None,  # No speed yet on first frame
                    "centroid": centroid,
                    "centroid_history": [centroid]  # Just current centroid
                })
                # Initialize track
                bbox = det[:4]
                centroid = compute_bbox_centroid(bbox)
                processing_state["vessel_tracks"][track_id] = {
                    "centroids": [centroid],
                    "velocities": [],
                    "speeds": [],
                    "accelerations": [],
                    "directions": [],
                    "bboxes": [bbox],
                    "frames": [processing_state["frame_count"]]
                }
                processing_state["active_track_ids"].add(track_id)
                processing_state["track_last_seen"][track_id] = processing_state["frame_count"]
            
            await websocket.send_json({
                "type": "frame",
                "frame": frame_base64,
                "frame_count": processing_state["frame_count"],
                "detections": detections_with_ids,
                "width": processing_state["width"],
                "height": processing_state["height"]
            })
        
        # Main processing loop
        while True:
            if processing_state["video_path"] is None or video_cap is None:
                await asyncio.sleep(0.1)
                continue
            
            # Handle auto-play state - play a few frames to gather detections
            if processing_state.get("auto_playing", False) and not processing_state["is_tracking"]:
                # Read next frame
                ret, frame = video_cap.read()
                if not ret:
                    # End of video during auto-play - stop and show last frame
                    processing_state["auto_playing"] = False
                    processing_state["is_paused"] = True
                    continue
                
                processing_state["frame_count"] += 1
                processing_state["current_frame"] = frame.copy()
                processing_state["auto_play_count"] += 1
                dt = 1.0 / processing_state["fps"]
                
                # Run YOLO detection
                detections = detect_vessels_with_yolo(frame, yolo_model, CONF_THRESHOLD)
                
                # Match detections to existing tracks or create new ones
                if processing_state["vessel_tracks"]:
                    matches, unmatched = match_detections_to_tracks(
                        detections, 
                        processing_state["vessel_tracks"], 
                        IOU_THRESHOLD
                    )
                    
                    # Update matched tracks
                    for track_id, det_idx in matches.items():
                        bbox = detections[det_idx][:4]
                        centroid = compute_bbox_centroid(bbox)
                        track_data = processing_state["vessel_tracks"][track_id]
                        track_data["bboxes"].append(bbox)
                        track_data["centroids"].append(centroid)
                        track_data["frames"].append(processing_state["frame_count"])
                        processing_state["track_last_seen"][track_id] = processing_state["frame_count"]
                    
                    # Create new tracks for unmatched detections
                    for det_idx in unmatched:
                        track_id = processing_state["next_track_id"]
                        processing_state["next_track_id"] += 1
                        bbox = detections[det_idx][:4]
                        centroid = compute_bbox_centroid(bbox)
                        processing_state["vessel_tracks"][track_id] = {
                            "centroids": [centroid],
                            "velocities": [],
                            "speeds": [],
                            "accelerations": [],
                            "directions": [],
                            "bboxes": [bbox],
                            "frames": [processing_state["frame_count"]]
                        }
                        processing_state["active_track_ids"].add(track_id)
                        processing_state["track_last_seen"][track_id] = processing_state["frame_count"]
                else:
                    # No existing tracks - assign new IDs
                    for det in detections:
                        track_id = processing_state["next_track_id"]
                        processing_state["next_track_id"] += 1
                        bbox = det[:4]
                        centroid = compute_bbox_centroid(bbox)
                        processing_state["vessel_tracks"][track_id] = {
                            "centroids": [centroid],
                            "velocities": [],
                            "speeds": [],
                            "accelerations": [],
                            "directions": [],
                            "bboxes": [bbox],
                            "frames": [processing_state["frame_count"]]
                        }
                        processing_state["active_track_ids"].add(track_id)
                        processing_state["track_last_seen"][track_id] = processing_state["frame_count"]
                
                # Build detections list for display
                detections_with_ids = []
                for track_id in processing_state["active_track_ids"]:
                    if track_id in processing_state["vessel_tracks"]:
                        track_data = processing_state["vessel_tracks"][track_id]
                        if track_data["bboxes"]:
                            last_bbox = track_data["bboxes"][-1]
                            centroid = track_data["centroids"][-1] if track_data["centroids"] else None
                            speed = track_data["speeds"][-1] if track_data["speeds"] else None
                            centroid_history = track_data["centroids"][-20:] if len(track_data["centroids"]) > 1 else (track_data["centroids"] if track_data["centroids"] else [])
                            detections_with_ids.append({
                                "track_id": track_id,
                                "bbox": last_bbox,
                                "confidence": 0.9,  # Approximate confidence
                                "speed": speed,
                                "centroid": centroid,
                                "centroid_history": centroid_history
                            })
                
                # Send frame
                frame_base64 = frame_to_base64(frame)
                await websocket.send_json({
                    "type": "frame",
                    "frame": frame_base64,
                    "frame_count": processing_state["frame_count"],
                    "detections": detections_with_ids,
                    "width": processing_state["width"],
                    "height": processing_state["height"]
                })
                
                # Check if we've played enough frames
                if processing_state["auto_play_count"] >= processing_state["auto_play_frames"]:
                    processing_state["auto_playing"] = False
                    processing_state["is_paused"] = True
                    print(f"✓ Auto-played {processing_state['auto_play_count']} frames. Ready for tagging.")
                    await websocket.send_json({
                        "type": "auto_play_complete",
                        "message": f"Played {processing_state['auto_play_count']} frames. Ready to tag vessels.",
                        "frame_count": processing_state["frame_count"]
                    })
                
                # Control frame rate during auto-play
                await asyncio.sleep(1.0 / processing_state["fps"])
                continue
            
            # Handle ready state (not paused, not tracking) - show current frame with all detections
            if not processing_state["is_paused"] and not processing_state["is_tracking"]:
                if processing_state["current_frame"] is not None:
                    frame = processing_state["current_frame"]
                    # Run YOLO to get detections
                    detections = detect_vessels_with_yolo(frame, yolo_model, CONF_THRESHOLD)
                    
                    # Match to existing tracks or create new ones
                    detections_with_ids = []
                    if processing_state["vessel_tracks"]:
                        matches, unmatched = match_detections_to_tracks(
                            detections, 
                            processing_state["vessel_tracks"], 
                            IOU_THRESHOLD
                        )
                        
                        # Add matched detections
                        for track_id, det_idx in matches.items():
                            track_data = processing_state["vessel_tracks"].get(track_id, {})
                            bbox = detections[det_idx][:4]
                            centroid = track_data.get("centroids", [])
                            centroid = centroid[-1] if centroid else compute_bbox_centroid(bbox)
                            speed = track_data.get("speeds", [])
                            speed = speed[-1] if speed else None
                            centroid_history = track_data.get("centroids", [])[-20:] if len(track_data.get("centroids", [])) > 1 else (track_data.get("centroids", []) if track_data.get("centroids") else [])
                            detections_with_ids.append({
                                "track_id": track_id,
                                "bbox": bbox,
                                "confidence": detections[det_idx][4],
                                "speed": speed,
                                "centroid": centroid,
                                "centroid_history": centroid_history
                            })
                        
                        # Create new tracks for unmatched detections
                        for det_idx in unmatched:
                            track_id = processing_state["next_track_id"]
                            processing_state["next_track_id"] += 1
                            bbox = detections[det_idx][:4]
                            centroid = compute_bbox_centroid(bbox)
                            processing_state["vessel_tracks"][track_id] = {
                                "centroids": [centroid],
                                "velocities": [],
                                "speeds": [],
                                "accelerations": [],
                                "directions": [],
                                "bboxes": [bbox],
                                "frames": [processing_state["frame_count"]]
                            }
                            processing_state["active_track_ids"].add(track_id)
                            processing_state["track_last_seen"][track_id] = processing_state["frame_count"]
                            detections_with_ids.append({
                                "track_id": track_id,
                                "bbox": bbox,
                                "confidence": detections[det_idx][4],
                                "speed": None,  # New track, no speed yet
                                "centroid": centroid,
                                "centroid_history": [centroid]
                            })
                    else:
                        # No existing tracks - assign new IDs
                        for det in detections:
                            track_id = processing_state["next_track_id"]
                            processing_state["next_track_id"] += 1
                            bbox = det[:4]
                            centroid = compute_bbox_centroid(bbox)
                            processing_state["vessel_tracks"][track_id] = {
                                "centroids": [centroid],
                                "velocities": [],
                                "speeds": [],
                                "accelerations": [],
                                "directions": [],
                                "bboxes": [bbox],
                                "frames": [processing_state["frame_count"]]
                            }
                            processing_state["active_track_ids"].add(track_id)
                            processing_state["track_last_seen"][track_id] = processing_state["frame_count"]
                            detections_with_ids.append({
                                "track_id": track_id,
                                "bbox": bbox,
                                "confidence": det[4],
                                "speed": None,  # New track, no speed yet
                                "centroid": centroid,
                                "centroid_history": [centroid]
                            })
                    
                    frame_base64 = frame_to_base64(frame)
                    await websocket.send_json({
                        "type": "frame",
                        "frame": frame_base64,
                        "frame_count": processing_state["frame_count"],
                        "detections": detections_with_ids,
                        "width": processing_state["width"],
                        "height": processing_state["height"]
                    })
                
                await asyncio.sleep(0.5)  # Update every 0.5s when ready
                continue
            
            # Handle pause state - send current frame with all detections for re-tagging
            if processing_state["is_paused"]:
                if processing_state["current_frame"] is not None:
                    frame = processing_state["current_frame"]
                    # Run YOLO to get fresh detections
                    detections = detect_vessels_with_yolo(frame, yolo_model, CONF_THRESHOLD)
                    
                    # Match to existing tracks to get track IDs
                    detections_with_ids = []
                    if processing_state["vessel_tracks"]:
                        matches, unmatched = match_detections_to_tracks(
                            detections, 
                            processing_state["vessel_tracks"], 
                            IOU_THRESHOLD
                        )
                        
                        # Add matched detections
                        for track_id, det_idx in matches.items():
                            track_data = processing_state["vessel_tracks"].get(track_id, {})
                            bbox = detections[det_idx][:4]
                            centroid = track_data.get("centroids", [])
                            centroid = centroid[-1] if centroid else compute_bbox_centroid(bbox)
                            speed = track_data.get("speeds", [])
                            speed = speed[-1] if speed else None
                            centroid_history = track_data.get("centroids", [])[-20:] if len(track_data.get("centroids", [])) > 1 else (track_data.get("centroids", []) if track_data.get("centroids") else [])
                            detections_with_ids.append({
                                "track_id": track_id,
                                "bbox": bbox,
                                "confidence": detections[det_idx][4],
                                "speed": speed,
                                "centroid": centroid,
                                "centroid_history": centroid_history
                            })
                        
                        # Create new tracks for unmatched detections
                        for det_idx in unmatched:
                            track_id = processing_state["next_track_id"]
                            processing_state["next_track_id"] += 1
                            bbox = detections[det_idx][:4]
                            centroid = compute_bbox_centroid(bbox)
                            processing_state["vessel_tracks"][track_id] = {
                                "centroids": [centroid],
                                "velocities": [],
                                "speeds": [],
                                "accelerations": [],
                                "directions": [],
                                "bboxes": [bbox],
                                "frames": [processing_state["frame_count"]]
                            }
                            processing_state["active_track_ids"].add(track_id)
                            processing_state["track_last_seen"][track_id] = processing_state["frame_count"]
                            detections_with_ids.append({
                                "track_id": track_id,
                                "bbox": bbox,
                                "confidence": detections[det_idx][4],
                                "speed": None,  # New track, no speed yet
                                "centroid": centroid,
                                "centroid_history": [centroid]
                            })
                    else:
                        # No existing tracks - assign new IDs
                        for det in detections:
                            track_id = processing_state["next_track_id"]
                            processing_state["next_track_id"] += 1
                            bbox = det[:4]
                            centroid = compute_bbox_centroid(bbox)
                            processing_state["vessel_tracks"][track_id] = {
                                "centroids": [centroid],
                                "velocities": [],
                                "speeds": [],
                                "accelerations": [],
                                "directions": [],
                                "bboxes": [bbox],
                                "frames": [processing_state["frame_count"]]
                            }
                            processing_state["active_track_ids"].add(track_id)
                            processing_state["track_last_seen"][track_id] = processing_state["frame_count"]
                            detections_with_ids.append({
                                "track_id": track_id,
                                "bbox": bbox,
                                "confidence": det[4],
                                "speed": None,  # New track, no speed yet
                                "centroid": centroid,
                                "centroid_history": [centroid]
                            })
                    
                    frame_base64 = frame_to_base64(frame)
                    await websocket.send_json({
                        "type": "frame",
                        "frame": frame_base64,
                        "frame_count": processing_state["frame_count"],
                        "detections": detections_with_ids,
                        "width": processing_state["width"],
                        "height": processing_state["height"]
                    })
                
                await asyncio.sleep(0.5)  # Update every 0.5s when paused
                continue
            
            # Handle tracking state
            if not processing_state["is_tracking"]:
                await asyncio.sleep(0.1)
                continue
            
            # Read next frame
            ret, frame = video_cap.read()
            if not ret:
                # End of video - stop tracking
                processing_state["is_tracking"] = False
                processing_state["is_paused"] = True
                print(f"\n✓ Video ended at frame {processing_state['frame_count']}. Tracking stopped.")
                await websocket.send_json({
                    "type": "video_ended",
                    "message": "Video playback completed",
                    "frame_count": processing_state["frame_count"]
                })
                # Stay in pause state
                continue
            
            processing_state["frame_count"] += 1
            processing_state["current_frame"] = frame.copy()
            dt = 1.0 / processing_state["fps"]
            
            # Run YOLO detection
            detections = detect_vessels_with_yolo(frame, yolo_model, CONF_THRESHOLD)
            
            # Check if we're in "all detections" mode (no selected_track_ids filter)
            # or "target vessels" mode (only track selected ones)
            if processing_state["selected_track_ids"]:
                # Target vessels mode - only track selected ones
                selected_tracks = {
                    tid: processing_state["vessel_tracks"][tid]
                    for tid in processing_state["selected_track_ids"]
                    if tid in processing_state["vessel_tracks"]
                }
                
                if selected_tracks and detections:
                    matches, unmatched = match_detections_to_tracks(detections, selected_tracks, IOU_THRESHOLD)
                else:
                    matches = {}
                    unmatched = list(range(len(detections)))
            else:
                # All detections mode - track all vessels
                all_tracks = processing_state["vessel_tracks"]
                if all_tracks and detections:
                    matches, unmatched = match_detections_to_tracks(detections, all_tracks, IOU_THRESHOLD)
                else:
                    matches = {}
                    unmatched = list(range(len(detections)))
            
            # Update matched tracks
            detections_with_ids = []
            for track_id, det_idx in matches.items():
                # In target vessels mode, only process selected tracks
                # In all detections mode, process all tracks
                if processing_state["selected_track_ids"] and track_id not in processing_state["selected_track_ids"]:
                    continue
                
                bbox = detections[det_idx][:4]
                centroid = compute_bbox_centroid(bbox)
                
                track_data = processing_state["vessel_tracks"][track_id]
                track_data["bboxes"].append(bbox)
                track_data["centroids"].append(centroid)
                track_data["frames"].append(processing_state["frame_count"])
                processing_state["track_last_seen"][track_id] = processing_state["frame_count"]
                
                # Compute motion (if we have previous centroid)
                if len(track_data["centroids"]) > 1:
                    prev_centroid = track_data["centroids"][-2]
                    curr_centroid = track_data["centroids"][-1]
                    
                    if validate_centroid_change(prev_centroid, curr_centroid, MAX_PIXEL_CHANGE):
                        raw_velocity = compute_velocity(prev_centroid, curr_centroid, dt)
                        velocity = smooth_velocity(raw_velocity, track_data["velocities"], 
                                                  SMOOTHING_FACTOR, MAX_VELOCITY_CHANGE_RATIO)
                        speed = compute_speed(velocity)
                        direction = compute_direction(velocity)
                        
                        track_data["velocities"].append(velocity)
                        track_data["speeds"].append(speed)
                        track_data["directions"].append(direction)
                        
                        # Compute acceleration
                        if len(track_data["velocities"]) > 1:
                            prev_velocity = track_data["velocities"][-2]
                            curr_velocity = track_data["velocities"][-1]
                            acceleration = compute_acceleration(prev_velocity, curr_velocity, dt)
                            track_data["accelerations"].append(acceleration)
                            
                            # Print motion calculations to terminal
                            print(f"\n{'='*70}")
                            print(f"TRACK ID {track_id} - Frame {processing_state['frame_count']}")
                            print(f"{'='*70}")
                            print(f"Centroid: ({curr_centroid[0]:.2f}, {curr_centroid[1]:.2f}) pixels")
                            print(f"Velocity: ({velocity[0]:.2f}, {velocity[1]:.2f}) px/s")
                            print(f"Speed: {speed:.2f} px/s")
                            print(f"Direction: {direction:.2f}°")
                            if acceleration:
                                accel_mag = math.sqrt(acceleration[0]**2 + acceleration[1]**2)
                                print(f"Acceleration: ({acceleration[0]:.2f}, {acceleration[1]:.2f}) px/s²")
                                print(f"Acceleration Magnitude: {accel_mag:.2f} px/s²")
                            print(f"{'='*70}\n")
                
                # Get speed and centroid history for this track
                speed = track_data["speeds"][-1] if track_data["speeds"] else None
                centroid_history = track_data["centroids"][-20:] if len(track_data["centroids"]) > 1 else []  # Last 20 centroids
                
                detections_with_ids.append({
                    "track_id": track_id,
                    "bbox": bbox,
                    "confidence": detections[det_idx][4],
                    "speed": speed,
                    "centroid": centroid,
                    "centroid_history": centroid_history
                })
            
            # Create new tracks for unmatched detections (only in all detections mode)
            if not processing_state["selected_track_ids"]:  # All detections mode
                for det_idx in unmatched:
                    detection = detections[det_idx]
                    track_id = processing_state["next_track_id"]
                    processing_state["next_track_id"] += 1
                    bbox = detection[:4]
                    centroid = compute_bbox_centroid(bbox)
                    processing_state["vessel_tracks"][track_id] = {
                        "centroids": [centroid],
                        "velocities": [],
                        "speeds": [],
                        "accelerations": [],
                        "directions": [],
                        "bboxes": [bbox],
                        "frames": [processing_state["frame_count"]]
                    }
                    processing_state["active_track_ids"].add(track_id)
                    processing_state["track_last_seen"][track_id] = processing_state["frame_count"]
                    detections_with_ids.append({
                        "track_id": track_id,
                        "bbox": bbox,
                        "confidence": detection[4],
                        "speed": None,  # New track, no speed yet
                        "centroid": centroid,
                        "centroid_history": [centroid]
                    })
            
            # Remove tracks that haven't been seen
            tracks_to_remove = []
            for track_id in list(processing_state["active_track_ids"]):
                # In target vessels mode, only manage selected tracks
                # In all detections mode, manage all tracks
                if processing_state["selected_track_ids"] and track_id not in processing_state["selected_track_ids"]:
                    continue
                if track_id in processing_state["track_last_seen"]:
                    frames_since_seen = processing_state["frame_count"] - processing_state["track_last_seen"][track_id]
                    if frames_since_seen > MAX_FRAMES_WITHOUT_DETECTION:
                        tracks_to_remove.append(track_id)
            
            for track_id in tracks_to_remove:
                processing_state["active_track_ids"].discard(track_id)
                if track_id in processing_state["vessel_tracks"]:
                    del processing_state["vessel_tracks"][track_id]
                if track_id in processing_state["track_last_seen"]:
                    del processing_state["track_last_seen"][track_id]
            
            # Filter detections based on mode
            if processing_state["selected_track_ids"]:
                # Target vessels mode - only show selected ones
                filtered_detections = [
                    det for det in detections_with_ids 
                    if det["track_id"] in processing_state["selected_track_ids"]
                ]
            else:
                # All detections mode - show all
                filtered_detections = detections_with_ids
            
            # Send frame and detections (only selected ones)
            frame_base64 = frame_to_base64(frame)
            await websocket.send_json({
                "type": "frame",
                "frame": frame_base64,
                "frame_count": processing_state["frame_count"],
                "detections": filtered_detections,
                "width": processing_state["width"],
                "height": processing_state["height"]
            })
            
            # Control frame rate
            await asyncio.sleep(1.0 / processing_state["fps"])
    
    except WebSocketDisconnect:
        active_connections.discard(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        active_connections.discard(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
