"""
Helper script to visualize and select bounding boxes for vessel detection.
This script helps you determine the correct bounding box coordinates.
"""

import cv2
import numpy as np
from pathlib import Path


def draw_bboxes_on_frame(frame, bboxes, labels=None):
    """
    Draw bounding boxes on a frame for visualization.
    
    Args:
        frame: Image frame (numpy array)
        bboxes: List of bounding boxes [[x1, y1, x2, y2], ...]
        labels: Optional list of labels for each bbox
    """
    frame_copy = frame.copy()
    
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0) if labels is None else ((i * 50) % 255, 255, 255)
        
        # Draw rectangle
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
        
        # Draw label if provided
        if labels:
            label = labels[i] if i < len(labels) else f"Vessel {i}"
            cv2.putText(frame_copy, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw coordinates
        coord_text = f"({x1},{y1}) to ({x2},{y2})"
        cv2.putText(frame_copy, coord_text, (x1, y2 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return frame_copy


def interactive_bbox_selection(frame):
    """
    Interactive bounding box selection using mouse clicks.
    Click and drag to select a bounding box. Press 'q' to finish.
    
    Returns:
        List of bounding boxes [[x1, y1, x2, y2], ...]
    """
    bboxes = []
    drawing = False
    start_point = None
    current_bbox = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start_point, current_bbox, bboxes
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
            current_bbox = None
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                current_bbox = [start_point[0], start_point[1], x, y]
        
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if start_point:
                # Ensure x1 < x2 and y1 < y2
                x1, y1 = min(start_point[0], x), min(start_point[1], y)
                x2, y2 = max(start_point[0], x), max(start_point[1], y)
                bboxes.append([x1, y1, x2, y2])
                print(f"Added bbox {len(bboxes)}: [{x1}, {y1}, {x2}, {y2}]")
                current_bbox = None
    
    window_name = "Select Bounding Boxes - Press 'q' to finish, 'r' to reset, 'c' to clear last"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    while True:
        display_frame = draw_bboxes_on_frame(frame, bboxes + ([current_bbox] if current_bbox else []))
        cv2.imshow(window_name, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            bboxes = []
            print("Reset all bounding boxes")
        elif key == ord('c'):
            if bboxes:
                removed = bboxes.pop()
                print(f"Removed last bbox: {removed}")
    
    cv2.destroyAllWindows()
    return bboxes


def visualize_existing_bboxes(video_path, bboxes):
    """
    Visualize existing bounding boxes on the first frame of the video.
    
    Args:
        video_path: Path to video file
        bboxes: List of bounding boxes to visualize
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not read first frame")
        return
    
    # Draw bounding boxes
    frame_with_bboxes = draw_bboxes_on_frame(frame, bboxes, 
                                            labels=[f"Vessel {i}" for i in range(len(bboxes))])
    
    # Display
    cv2.imshow("Bounding Boxes Preview - Press any key to close", frame_with_bboxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Print coordinates
    print("\nBounding box coordinates:")
    print("initial_bboxes = [")
    for bbox in bboxes:
        print(f"    {bbox},")
    print("]")
    
    return frame_with_bboxes


if __name__ == "__main__":
    import sys
    
    video_path = Path("media/boat_shipping1.mp4")
    
    if not video_path.exists():
        print(f"Error: Video not found at {video_path}")
        sys.exit(1)
    
    # Extract first frame
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not read first frame")
        sys.exit(1)
    
    print("=" * 60)
    print("Interactive Bounding Box Selection")
    print("=" * 60)
    print("Instructions:")
    print("  - Click and drag to select a bounding box around a vessel")
    print("  - Press 'q' to finish and save bounding boxes")
    print("  - Press 'r' to reset all bounding boxes")
    print("  - Press 'c' to clear the last bounding box")
    print("  - You can select multiple vessels")
    print("=" * 60)
    
    # Interactive selection
    bboxes = interactive_bbox_selection(frame)
    
    if bboxes:
        print(f"\nSelected {len(bboxes)} bounding box(es):")
        for i, bbox in enumerate(bboxes):
            print(f"  Vessel {i}: {bbox}")
        
        # Show final visualization
        print("\nShowing final visualization...")
        visualize_existing_bboxes(video_path, bboxes)
        
        # Save to file
        output_file = Path("vessel_segments") / "selected_bboxes.txt"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("# Bounding boxes for vessel detection\n")
            f.write("# Format: [[x1, y1, x2, y2], ...]\n")
            f.write("# Copy these into vessel_detection_sam2.py as initial_bboxes\n\n")
            f.write("initial_bboxes = [\n")
            for bbox in bboxes:
                f.write(f"    {bbox},\n")
            f.write("]\n")
        
        print(f"\nBounding boxes saved to: {output_file}")
        print("You can copy the coordinates into vessel_detection_sam2.py")
    else:
        print("\nNo bounding boxes selected.")
