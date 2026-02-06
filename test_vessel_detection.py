"""
Test/Inference script for trained YOLOv8 vessel detection model
Runs detection on videos in /media directory and saves annotated results
"""

import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from ultralytics import YOLO
import argparse


def process_video(model, video_path, output_dir, conf_threshold=0.25, save_video=True, save_frames=False):
    """
    Process a single video with the trained YOLO model.
    
    Args:
        model: Loaded YOLO model
        video_path: Path to input video
        output_dir: Directory to save results
        conf_threshold: Confidence threshold for detections
        save_video: Whether to save annotated video
        save_frames: Whether to save individual frames with detections
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Processing: {video_path.name}")
    print(f"{'='*60}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup video writer if saving video
    video_writer = None
    if save_video:
        output_video_path = output_dir / f"{video_path.stem}_detections.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        print(f"Output video: {output_video_path}")
    
    # Storage for detection results
    detection_results = {
        "video_path": str(video_path),
        "fps": fps,
        "resolution": [width, height],
        "total_frames": total_frames,
        "conf_threshold": conf_threshold,
        "detections": []
    }
    
    frame_count = 0
    total_detections = 0
    
    print("Processing frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO detection
        results = model(frame, conf=conf_threshold, verbose=False)
        
        # Process detections
        frame_detections = []
        annotated_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[class_id]
                
                # Store detection
                frame_detections.append({
                    "frame": frame_count,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name
                })
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, 
                            (int(x1), int(y1)), (int(x2), int(y2)), 
                            (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name} {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame,
                            (int(x1), int(y1) - label_size[1] - 10),
                            (int(x1) + label_size[0], int(y1)),
                            (0, 255, 0), -1)
                cv2.putText(annotated_frame, label,
                          (int(x1), int(y1) - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                total_detections += 1
        
        # Store frame detections
        if frame_detections:
            detection_results["detections"].append({
                "frame": frame_count,
                "count": len(frame_detections),
                "detections": frame_detections
            })
        
        # Save annotated frame if requested
        if save_frames and frame_detections:
            frame_path = output_dir / f"{video_path.stem}_frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(frame_path), annotated_frame)
        
        # Write frame to output video
        if video_writer:
            video_writer.write(annotated_frame)
        
        frame_count += 1
        
        # Progress update every 30 frames
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Progress: {frame_count}/{total_frames} frames ({progress:.1f}%) - {total_detections} detections")
    
    # Cleanup
    cap.release()
    if video_writer:
        video_writer.release()
    
    # Save detection results as JSON
    json_path = output_dir / f"{video_path.stem}_detections.json"
    with open(json_path, 'w') as f:
        json.dump(detection_results, f, indent=2)
    
    print(f"\n✓ Completed: {video_path.name}")
    print(f"  Total frames: {frame_count}")
    print(f"  Total detections: {total_detections}")
    print(f"  Frames with detections: {len(detection_results['detections'])}")
    print(f"  Results saved to: {json_path}")
    if save_video:
        print(f"  Video saved to: {output_dir / f'{video_path.stem}_detections.mp4'}")
    
    return detection_results


def main():
    parser = argparse.ArgumentParser(description="Test trained YOLOv8 vessel detection model on videos")
    parser.add_argument("--model", type=str, 
                       default="runs/detect/runs/detect/vessel_detection_yolov8n11/weights/last.pt",
                       help="Path to trained model (default: last.pt)")
    parser.add_argument("--video-dir", type=str, default="media",
                       help="Directory containing videos (default: media)")
    parser.add_argument("--output-dir", type=str, default="test_results",
                       help="Output directory for results (default: test_results)")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Confidence threshold (default: 0.25)")
    parser.add_argument("--save-video", action="store_true", default=True,
                       help="Save annotated video (default: True)")
    parser.add_argument("--save-frames", action="store_true",
                       help="Save individual frames with detections")
    parser.add_argument("--video", type=str, default=None,
                       help="Process single video file (optional)")
    
    args = parser.parse_args()
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
    
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    print(f"✓ Model loaded successfully")
    print(f"  Model classes: {list(model.names.values())}")
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = output_dir / f"test_session_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {session_dir}")
    
    # Get video files
    video_dir = Path(args.video_dir)
    if args.video:
        video_files = [Path(args.video)]
    else:
        video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi")) + list(video_dir.glob("*.mov"))
    
    if not video_files:
        print(f"Error: No video files found in {video_dir}")
        return
    
    print(f"\nFound {len(video_files)} video(s) to process:")
    for vf in video_files:
        print(f"  - {vf.name}")
    
    # Process each video
    all_results = []
    for video_file in video_files:
        results = process_video(
            model=model,
            video_path=video_file,
            output_dir=session_dir / video_file.stem,
            conf_threshold=args.conf,
            save_video=args.save_video,
            save_frames=args.save_frames
        )
        if results:
            all_results.append(results)
    
    # Save summary
    summary = {
        "test_timestamp": timestamp,
        "model_path": str(model_path),
        "conf_threshold": args.conf,
        "total_videos": len(video_files),
        "processed_videos": len(all_results),
        "results": all_results
    }
    
    summary_path = session_dir / "test_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    print(f"Total videos processed: {len(all_results)}")
    print(f"Summary saved to: {summary_path}")
    print(f"All results saved to: {session_dir}")


if __name__ == "__main__":
    main()
