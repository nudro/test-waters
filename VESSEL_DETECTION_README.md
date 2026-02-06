# Vessel Detection System - Quick Start

A clean, modern vessel detection and tracking system with React GUI and FastAPI backend.

## Architecture

- **Backend**: FastAPI server with WebSocket for real-time frame streaming
- **Frontend**: React + Vite for modern, responsive UI
- **Detection**: YOLOv8n model (`models/yolov8n_vessels.pt`)
- **No OpenCV GUI**: Uses React/Canvas instead - no X11/display dependencies

## Quick Start

### 1. Install Backend Dependencies

```bash
pip install -r requirements_vessel_detection.txt
```

### 2. Install Frontend Dependencies

```bash
cd vessel-detection-gui
npm install
```

### 3. Start Backend Server

```bash
python vessel_detection_server.py
```

Server runs on `http://localhost:8000`

### 4. Start Frontend (in a new terminal)

```bash
cd vessel-detection-gui
npm run dev
```

Frontend runs on `http://localhost:3000`

### 5. Open Browser

Navigate to `http://localhost:3000`

## Usage Workflow

1. **Load Video**: Select a video from the sidebar (searches `/media` directory)
2. **First Frame**: System automatically runs YOLO and shows all detections with track IDs
3. **Tag Vessels**: Click on vessel bounding boxes to select them (they turn **green**)
4. **Confirm**: Click **TAG** button to confirm your selection
5. **Track**: Click **TRACK** button to start tracking selected vessels
6. **View Motion**: Motion calculations (centroid, velocity, speed, acceleration, direction) stream to terminal
7. **Pause**: Click **PAUSE** at any time to stop and re-tag different vessels
8. **Re-tag**: When paused, click different bboxes and click **TAG** again, then **TRACK**

## Features

✅ **No OpenCV GUI** - Pure React/Canvas rendering  
✅ **WebSocket Streaming** - Real-time frame updates  
✅ **Interactive Tagging** - Click bboxes to select (green highlight)  
✅ **Motion Calculations** - Centroid, velocity, speed, acceleration, direction  
✅ **Terminal Output** - Motion data streams to backend terminal  
✅ **Pause & Re-tag** - Stop tracking anytime to select different vessels  
✅ **Video Search** - Automatically searches `/media` directory  

## File Structure

```
.
├── vessel_detection_server.py      # FastAPI backend
├── requirements_vessel_detection.txt
├── vessel-detection-gui/          # React frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── VideoDisplay.jsx    # Canvas with bbox overlay
│   │   │   ├── ControlPanel.jsx   # TAG/TRACK/PAUSE buttons
│   │   │   └── VideoSelector.jsx  # Video list
│   │   └── App.jsx                # Main app
│   └── package.json
└── media/                          # Video files directory
```

## API Endpoints

- `GET /api/videos` - List available videos
- `POST /api/load-video` - Load a video file
- `POST /api/tag` - Tag selected track IDs
- `POST /api/track` - Start tracking
- `POST /api/pause` - Pause tracking
- `WS /ws` - WebSocket for frame streaming

## Motion Calculations

Motion calculations are printed to the terminal where the backend is running:

```
======================================================================
TRACK ID 0 - Frame 150
======================================================================
Centroid: (640.50, 360.25) pixels
Velocity: (2.30, -1.45) px/s
Speed: 2.72 px/s
Direction: -32.15°
Acceleration: (0.10, -0.05) px/s²
Acceleration Magnitude: 0.11 px/s²
======================================================================
```

## Troubleshooting

### "YOLO model not found"
- Ensure `models/yolov8n_vessels.pt` exists
- Check the model path in `vessel_detection_server.py`

### "No videos found"
- Check that `/media` directory exists and contains video files
- Or place videos in project `media/` directory

### WebSocket connection failed
- Ensure backend is running on port 8000
- Check browser console for connection errors

### Frontend not loading
- Ensure you're accessing `http://localhost:3000` (not 8000)
- Check that `npm install` completed successfully

## Why This Approach is Cleaner

1. **Separation of Concerns**: Backend handles processing, frontend handles UI
2. **No Display Dependencies**: No X11/OpenCV GUI - works headless
3. **Remote Access**: Access from any browser, anywhere
4. **Modern Stack**: React is maintainable and extensible
5. **Real-time**: WebSocket enables low-latency streaming
6. **Simpler**: No complex OpenCV window management or mouse callbacks

