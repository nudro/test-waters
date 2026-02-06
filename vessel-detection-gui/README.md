# Vessel Detection GUI

A modern React-based GUI for vessel detection and tracking using YOLO.

## Features

- **Video Selection**: Browse and select videos from `/media` directory
- **Interactive Tagging**: Click on vessel bounding boxes to select them (they turn green)
- **Real-time Tracking**: Track selected vessels with motion calculations
- **Motion Calculations**: View real-time motion data (centroid, velocity, speed, acceleration, direction) in terminal
- **Pause & Re-tag**: Pause tracking at any time to select different vessels

## Setup

### 1. Install Backend Dependencies

```bash
pip install fastapi uvicorn websockets opencv-python ultralytics numpy
```

### 2. Install Frontend Dependencies

```bash
cd vessel-detection-gui
npm install
```

### 3. Start the Backend Server

In one terminal:

```bash
python vessel_detection_server.py
```

The server will start on `http://localhost:8000`

### 4. Start the Frontend

In another terminal:

```bash
cd vessel-detection-gui
npm run dev
```

The frontend will start on `http://localhost:3000`

## Usage

1. **Load Video**: Select a video from the list in the sidebar
2. **Tag Vessels**: Click on vessel bounding boxes to select them (they will turn green)
3. **Confirm Selection**: Click the **TAG** button to confirm your selection
4. **Start Tracking**: Click the **TRACK** button to start tracking selected vessels
5. **View Results**: Motion calculations will stream to the terminal where the backend is running
6. **Pause & Re-tag**: Click **PAUSE** at any time to stop tracking and select different vessels

## Architecture

- **Backend**: FastAPI server with WebSocket support for real-time frame streaming
- **Frontend**: React + Vite for modern, responsive UI
- **Detection**: YOLOv8n model for vessel detection
- **Tracking**: IoU-based matching with motion calculations

## File Structure

```
vessel-detection-gui/
├── src/
│   ├── components/
│   │   ├── VideoDisplay.jsx    # Canvas-based video display with bbox overlay
│   │   ├── ControlPanel.jsx   # TAG/TRACK/PAUSE controls
│   │   └── VideoSelector.jsx  # Video selection from /media
│   ├── App.jsx                # Main app component
│   └── main.jsx              # Entry point
├── package.json
└── vite.config.js
```

