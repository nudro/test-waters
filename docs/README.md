# YOLO+SAM

## Overview

This document outlines the plan for combining YOLOv8 vessel detection with SAM2 segmentation and tracking to create an automated pipeline for vessel detection, tracking, and motion estimation.

## Step 1: Train Custom YOLOv8 Model

### Dataset Sources

Train YOLOv8m on the following datasets:

1. **`datasets/archive/`**
   - Use directories: `/v1` through `/v7`
   - **Ignore**: `/v8`

2. **`datasets/Maritime YOLO.v1i.yolov8/`**
   - Include this dataset in training

### Expected Output

A custom trained YOLOv8 model (`models/yolov8m_custom.pt` or similar) that can detect vessels in maritime video frames.

## Step 2: Integration with SAM2

### Workflow: YOLO + SAM2 for Vessel Tracking and Motion Estimation

### Detection Phase (YOLO)

- YOLO runs on each frame to detect vessels
- Outputs bounding boxes for each detected vessel
- These bounding boxes serve as prompts for SAM2

### Segmentation & Tracking Phase (SAM2)

- YOLO bounding boxes are fed to SAM2 as prompts
- SAM2 generates pixel-level segmentation masks for each vessel
- SAM2 tracks vessels across frames using its memory mechanism
- Handles occlusions, reappearances, and appearance changes

## Step 3: Motion Estimation Capabilities

### What SAM2 Provides

1. **Temporal Tracking**: Maintains vessel identity across frames, even with occlusions
2. **Pixel-Level Masks**: Precise vessel boundaries for accurate motion analysis
3. **Memory Mechanism**: Tracks vessels that temporarily disappear and reappear

### Motion Metrics You Can Compute

From SAM2 mask outputs, you can calculate:

1. **Center of Mass**: Compute mask centroid per frame → track position over time
2. **Velocity Vectors**: Difference in center positions between frames
3. **Trajectory Paths**: Full path of each vessel through the video
4. **Speed Estimation**: Distance traveled / time (using video FPS)
5. **Direction Changes**: Angle changes in movement vectors
6. **Area Changes**: Mask area changes (useful for scale/distance estimation)

## Implementation Approaches

### Option A: YOLO-First Approach (Recommended)

1. Run YOLO on each frame → get bounding boxes
2. Feed those boxes to SAM2 as prompts → get masks
3. SAM2 tracks the same vessels across frames
4. Extract motion data from mask positions/centers over time

**Advantages:**
- Fully automated
- No manual intervention needed
- YOLO handles detection, SAM2 handles tracking

### Option B: Hybrid Approach

1. Use YOLO for initial detection in first frame
2. Initialize SAM2 with those detections
3. Let SAM2 track through the video (can handle new detections too)
4. Periodically re-run YOLO to catch new vessels or refine detections

**Advantages:**
- More efficient (YOLO not needed every frame)
- SAM2 memory handles most tracking
- YOLO used for validation/refinement

## Advantages of YOLO + SAM2 Combination

- **YOLO**: Fast, automatic detection, no manual prompts needed
- **SAM2**: Precise segmentation, robust tracking, handles occlusions
- **Together**: Automated pipeline from detection → segmentation → tracking → motion analysis

## Motion Estimation Outputs

The combined system can provide:

- **Per-vessel trajectories** (x, y coordinates over time)
- **Velocity vectors** (speed and direction)
- **Acceleration/deceleration patterns**
- **Collision risk analysis** (if multiple vessels)
- **Path prediction** (extrapolate trajectories)

## Key Concept

The key to this approach is using YOLO detections as prompts for SAM2, then analyzing the mask positions/centers over time to compute motion metrics. This creates a fully automated pipeline that requires no manual bounding box selection per video.

