"""
Interactive Track ID Assignment Tool
Shows first frame with YOLO detections and allows user to click on vessels to assign track IDs.
"""

import cv2
import numpy as np
from pathlib import Path
import json
from ultralytics import YOLO
import argparse


def detect_vessels_with_yolo(frame, yolo_model, conf_threshold=0.25):
    """
    Use YOLO to detect vessels in a frame and return bounding boxes.
    
    Returns:
        List of [x1, y1, x2, y2, confidence] for each detection
    """
    results = yolo_model(frame, conf=conf_threshold, verbose=False)
    bboxes = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            bboxes.append([int(x1), int(y1), int(x2), int(y2), confidence])
    
    return bboxes


def point_in_bbox(point, bbox):
    """Check if a point is inside a bounding box."""
    x, y = point
    x1, y1, x2, y2 = bbox[:4]
    return x1 <= x <= x2 and y1 <= y <= y2


def assign_track_ids_interactive(video_path, yolo_model_path="models/yolov8n_vessels.pt", 
                                 output_dir="vessel_tracking_results", conf_threshold=0.25):
    """
    Interactive tool to assign track IDs to vessels in the first frame.
    
    Args:
        video_path: Path to input video
        yolo_model_path: Path to YOLO model
        output_dir: Directory to save mapping file
        conf_threshold: YOLO confidence threshold
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Interactive Track ID Assignment Tool")
    print(f"{'='*70}")
    print(f"Video: {video_path.name}")
    print(f"YOLO Model: {yolo_model_path}")
    print(f"{'='*70}\n")
    
    # Load YOLO model
    print("Loading YOLO model...")
    yolo_model = YOLO(yolo_model_path)
    print(f"✓ YOLO model loaded")
    
    # Open video and get first frame
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Could not read first frame from video")
    
    cap.release()
    
    # Get video info
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video resolution: {width}x{height}")
    print(f"FPS: {fps}\n")
    
    # Run YOLO detection on first frame
    print("Detecting vessels in first frame...")
    detections = detect_vessels_with_yolo(frame, yolo_model, conf_threshold)
    print(f"✓ Found {len(detections)} vessel(s)\n")
    
    if len(detections) == 0:
        print("No vessels detected in first frame. Exiting.")
        return None
    
    # Create a copy for drawing
    display_frame = frame.copy()
    
    # Color palette for different track IDs
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
        (0, 128, 255), (128, 255, 0), (255, 192, 203), (128, 128, 0),
        (0, 128, 128), (128, 0, 0), (0, 0, 128), (192, 192, 192),
        (128, 128, 128), (64, 64, 64), (255, 128, 0), (0, 255, 128)
    ]
    
    # Track assignments: {detection_index: track_id}
    assignments = {}  # {detection_index: track_id}
    next_track_id = 0
    unassigned_detections = set(range(len(detections)))
    
    
    def draw_assignments(img, detections, assignments, colors):
        """Draw detections with assigned track IDs."""
        for det_idx, bbox in enumerate(detections):
            x1, y1, x2, y2 = bbox[:4]
            confidence = bbox[4]
            
            if det_idx in assignments:
                # Assigned - show with color and ID
                track_id = assignments[det_idx]
                color = colors[track_id % len(colors)]
                label = f"ID: {track_id}"
                thickness = 3
            else:
                # Unassigned - show in gray
                color = (128, 128, 128)
                label = f"Unassigned (Click to assign)"
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, 
                         (x1 - 2, y1 - text_height - 10), 
                         (x1 + text_width + 5, y1 + 5), 
                         color, -1)
            cv2.putText(img, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw confidence
            conf_text = f"Conf: {confidence:.2f}"
            cv2.putText(img, conf_text, (x1, y2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Initial draw
    draw_assignments(display_frame, detections, assignments, colors)
    
    # Resize for display (75% of original)
    display_scale = 0.75
    display_width = int(display_frame.shape[1] * display_scale)
    display_height = int(display_frame.shape[0] * display_scale)
    display_frame_resized = cv2.resize(display_frame, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
    
    # Create window and set mouse callback
    # Pass display_scale to callback via closure
    def mouse_callback_with_scale(event, x, y, flags, param):
        nonlocal next_track_id, assignments, unassigned_detections, display_frame, display_frame_resized, display_width, display_height
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert click coordinates from display (resized) coordinates to original frame coordinates
            # The display is scaled by display_scale (0.75), so we need to scale back up
            original_x = int(x / display_scale)
            original_y = int(y / display_scale)
            
            # Find which detection was clicked (using original coordinates)
            clicked_detection = None
            for det_idx, bbox in enumerate(detections):
                if point_in_bbox((original_x, original_y), bbox):
                    clicked_detection = det_idx
                    break
            
            if clicked_detection is not None:
                if clicked_detection in assignments:
                    # Already assigned - remove assignment
                    del assignments[clicked_detection]
                    unassigned_detections.add(clicked_detection)
                    print(f"  Removed assignment from detection {clicked_detection}")
                else:
                    # Assign new track ID
                    track_id = next_track_id
                    assignments[clicked_detection] = track_id
                    unassigned_detections.discard(clicked_detection)
                    next_track_id += 1
                    print(f"  Assigned detection {clicked_detection} → TRACK ID {track_id}")
                
                # Redraw frame
                display_frame = frame.copy()
                draw_assignments(display_frame, detections, assignments, colors)
                display_frame_resized = cv2.resize(display_frame, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
                cv2.imshow("Assign Track IDs", display_frame_resized)
    
    cv2.namedWindow("Assign Track IDs", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Assign Track IDs", mouse_callback_with_scale)
    
    print(f"\n{'='*70}")
    print(f"INSTRUCTIONS:")
    print(f"  - Click on a vessel to assign it a Track ID")
    print(f"  - Click again to remove assignment")
    print(f"  - Press 's' to save and exit")
    print(f"  - Press 'q' to quit without saving")
    print(f"  - Press 'r' to reset all assignments")
    print(f"{'='*70}\n")
    
    cv2.imshow("Assign Track IDs", display_frame_resized)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            # Save assignments
            if len(assignments) == 0:
                print("\n⚠️  No assignments made. Nothing to save.")
                response = input("  Quit anyway? (y/n): ")
                if response.lower() != 'y':
                    continue
            
            # Create mapping file
            mapping = {}
            for det_idx, track_id in assignments.items():
                bbox = detections[det_idx]
                x1, y1, x2, y2 = bbox[:4]
                centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                
                mapping[str(track_id)] = {
                    "first_frame": 1,
                    "first_position": [float(x1), float(y1)],
                    "first_centroid": [float(centroid[0]), float(centroid[1])],
                    "first_size": [float(bbox_width), float(bbox_height)],
                    "first_bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "detection_index": int(det_idx),
                    "confidence": float(bbox[4]),
                    "manual_assignment": True
                }
            
            # Save mapping file
            mapping_file = output_dir / f"{video_path.stem}_track_id_mapping.json"
            with open(mapping_file, 'w') as f:
                json.dump(mapping, f, indent=2)
            
            print(f"\n✓ Saved {len(assignments)} track ID assignment(s)")
            print(f"✓ Mapping file: {mapping_file}")
            print(f"\n  Assigned Track IDs:")
            for det_idx, track_id in sorted(assignments.items()):
                bbox = detections[det_idx]
                print(f"    TRACK ID {track_id}: Detection {det_idx} at ({bbox[0]:.0f}, {bbox[1]:.0f})")
            
            cv2.destroyAllWindows()
            return mapping_file
            
        elif key == ord('q'):
            print("\n⚠️  Quitting without saving.")
            cv2.destroyAllWindows()
            return None
            
        elif key == ord('r'):
            # Reset all assignments
            assignments.clear()
            unassigned_detections = set(range(len(detections)))
            next_track_id = 0
            display_frame = frame.copy()
            draw_assignments(display_frame, detections, assignments, colors)
            display_frame_resized = cv2.resize(display_frame, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Assign Track IDs", display_frame_resized)
            print(f"\n  Reset all assignments")
        
        # Update display if window is resized
        if cv2.getWindowProperty("Assign Track IDs", cv2.WND_PROP_VISIBLE) < 1:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Track ID Assignment Tool")
    parser.add_argument("--video", type=str, required=True,
                       help="Path to input video")
    parser.add_argument("--yolo-model", type=str, default="models/yolov8n_vessels.pt",
                       help="Path to YOLO model (default: models/yolov8n_vessels.pt)")
    parser.add_argument("--output-dir", type=str, default="vessel_tracking_results",
                       help="Output directory (default: vessel_tracking_results)")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="YOLO confidence threshold (default: 0.25)")
    
    args = parser.parse_args()
    
    assign_track_ids_interactive(
        video_path=args.video,
        yolo_model_path=args.yolo_model,
        output_dir=args.output_dir,
        conf_threshold=args.conf
    )
