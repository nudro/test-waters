# CV Agent - Vessel Tracking Assistant

A LangChain-based ReAct agent that provides natural language interface to vessel detection and tracking functions.

## Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements_agent.txt
   ```

2. **Verify API Key:**
   - Ensure your OpenAI API key is at: `/home/nudro/Documents/keys/maritime-oai`
   - The key should be a single line with no extra whitespace

3. **Run the Agent:**
   ```bash
   python main.py
   ```

## Usage Examples

### Process a Video
```
You: Process video /media/nudro/USB DISK/all_shipping.mp4
```

### Get Vessel Motion Data
```
You: Get motion data for track-id 2 from vessel_tracking_results/all_shipping_tracking_results.json
```

### List All Tracked Vessels
```
You: List all tracked vessels in vessel_tracking_results/all_shipping_tracking_results.json
```

### Calculate Velocity
```
You: What is the velocity of track-id 1?
```

### Get Trajectory
```
You: Show me the trajectory of track-id 0
```

### Assign Track IDs (GUI)
```
You: Open GUI to assign track IDs for /media/nudro/USB DISK/all_shipping.mp4
```

## Available Tools

The agent has access to these tools:

1. **Motion Calculation Tools:**
   - `calculate_centroid` - Calculate bounding box center
   - `calculate_velocity` - Calculate velocity from positions
   - `calculate_speed_from_velocity` - Calculate speed magnitude
   - `calculate_acceleration` - Calculate acceleration vector
   - `calculate_direction` - Calculate direction angle

2. **Vessel Query Tools:**
   - `get_vessel_motion_data` - Get complete motion data for a track ID
   - `list_tracked_vessels` - List all available track IDs
   - `get_vessel_trajectory` - Get trajectory path points

3. **Video Processing Tools:**
   - `process_video_tracking` - Run full detection and tracking pipeline
   - `assign_track_ids_gui` - Open GUI for manual track ID assignment

## Architecture

```
main.py (Chat Interface)
    ↓
agent.py (ReAct Agent)
    ↓
tools.py (LangChain Tools)
    ↓
vessel_tracking_pipeline.py (Core Functions)
```

## Notes

- The agent uses GPT-4 by default (configurable in `agent.py`)
- Temperature is set to 0.1 for deterministic tool calling
- All tools return JSON-serializable dictionaries
- Results are saved to `vessel_tracking_results/` directory

## Troubleshooting

**Agent initialization fails:**
- Check API key file exists and is readable
- Verify OpenAI API key is valid
- Ensure all dependencies are installed

**Tool execution errors:**
- Check that video files exist at specified paths
- Verify YOLO model exists at `models/yolov8n_vessels.pt`
- Ensure tracking results JSON files exist before querying

**Import errors:**
- Make sure you're in the `maritime_detection` directory
- Verify `vessel_tracking_pipeline.py` is in the same directory
