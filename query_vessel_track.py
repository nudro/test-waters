"""
Utility script to query and analyze specific vessels by TRACK ID
from vessel tracking results
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_tracking_results(json_path):
    """Load tracking results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def get_vessel_motion(tracking_data, track_id):
    """
    Get motion data for a specific vessel by TRACK ID.
    
    Args:
        tracking_data: Full tracking results dictionary
        track_id: TRACK ID of the vessel
    
    Returns:
        Dictionary with motion data for this vessel
    """
    if track_id not in tracking_data["vessels"]:
        return None
    
    vessel = tracking_data["vessels"][track_id]
    
    # Get current (latest) values
    current_data = {
        "track_id": track_id,
        "current_position": vessel["centroids"][-1] if vessel["centroids"] else None,
        "current_velocity": vessel["velocities"][-1] if vessel["velocities"] else None,
        "current_speed": vessel["speeds"][-1] if vessel["speeds"] else None,
        "current_acceleration": vessel["accelerations"][-1] if vessel["accelerations"] else None,
        "current_direction": vessel["directions"][-1] if vessel["directions"] else None,
        "trajectory": vessel["centroids"],
        "velocity_history": vessel["velocities"],
        "acceleration_history": vessel["accelerations"],
        "speed_history": vessel["speeds"],
        "direction_history": vessel["directions"],
        "frames": vessel["frames"],
        "timestamps": vessel["timestamps"],
        "summary": {
            "avg_speed": vessel.get("avg_speed"),
            "max_speed": vessel.get("max_speed"),
            "avg_acceleration": vessel.get("avg_acceleration"),
            "total_frames": vessel.get("total_frames"),
        }
    }
    
    return current_data


def list_all_vessels(tracking_data):
    """List all available vessels with their TRACK IDs and summary info."""
    print(f"\n{'='*70}")
    print(f"Available Vessels (TRACK IDs)")
    print(f"{'='*70}")
    print(f"{'TRACK ID':<12} {'Frames':<10} {'Avg Speed':<15} {'Max Speed':<15}")
    print(f"{'-'*70}")
    
    for track_id in sorted(tracking_data["vessels"].keys(), key=lambda x: int(x)):
        vessel = tracking_data["vessels"][track_id]
        track_id_str = str(track_id)
        frames = vessel.get("total_frames", len(vessel.get("frames", [])))
        avg_speed = vessel.get("avg_speed")
        max_speed = vessel.get("max_speed")
        
        avg_speed_str = f"{avg_speed:.2f} px/s" if avg_speed else "N/A"
        max_speed_str = f"{max_speed:.2f} px/s" if max_speed else "N/A"
        
        print(f"{track_id_str:<12} {frames:<10} {avg_speed_str:<15} {max_speed_str:<15}")
    
    print(f"{'='*70}\n")


def print_vessel_details(vessel_motion):
    """Print detailed information about a specific vessel."""
    if vessel_motion is None:
        print("Error: Vessel not found")
        return
    
    print(f"\n{'='*70}")
    print(f"Vessel TRACK ID: {vessel_motion['track_id']}")
    print(f"{'='*70}")
    
    # Current state
    print(f"\nCurrent State:")
    if vessel_motion['current_position']:
        print(f"  Position: ({vessel_motion['current_position'][0]:.2f}, {vessel_motion['current_position'][1]:.2f})")
    if vessel_motion['current_velocity']:
        vx, vy = vessel_motion['current_velocity']
        print(f"  Velocity: ({vx:.2f}, {vy:.2f}) px/s")
    if vessel_motion['current_speed']:
        print(f"  Speed: {vessel_motion['current_speed']:.2f} px/s")
    if vessel_motion['current_direction']:
        print(f"  Direction: {vessel_motion['current_direction']:.2f}°")
    if vessel_motion['current_acceleration']:
        ax, ay = vessel_motion['current_acceleration']
        accel_mag = np.sqrt(ax**2 + ay**2)
        print(f"  Acceleration: ({ax:.2f}, {ay:.2f}) px/s² (magnitude: {accel_mag:.2f})")
    
    # Summary statistics
    print(f"\nSummary Statistics:")
    summary = vessel_motion['summary']
    print(f"  Total Frames Tracked: {summary['total_frames']}")
    if summary['avg_speed']:
        print(f"  Average Speed: {summary['avg_speed']:.2f} px/s")
    if summary['max_speed']:
        print(f"  Maximum Speed: {summary['max_speed']:.2f} px/s")
    if summary['avg_acceleration']:
        print(f"  Average Acceleration: {summary['avg_acceleration']:.2f} px/s²")
    
    # Trajectory info
    if vessel_motion['trajectory']:
        print(f"\nTrajectory:")
        print(f"  Total Points: {len(vessel_motion['trajectory'])}")
        start_pos = vessel_motion['trajectory'][0]
        end_pos = vessel_motion['trajectory'][-1]
        distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        print(f"  Start Position: ({start_pos[0]:.2f}, {start_pos[1]:.2f})")
        print(f"  End Position: ({end_pos[0]:.2f}, {end_pos[1]:.2f})")
        print(f"  Total Distance Traveled: {distance:.2f} pixels")
    
    print(f"{'='*70}\n")


def plot_vessel_trajectory(tracking_data, track_id, output_path=None):
    """Plot trajectory and motion metrics for a specific vessel."""
    vessel_motion = get_vessel_motion(tracking_data, track_id)
    if vessel_motion is None:
        print(f"Error: Vessel TRACK ID {track_id} not found")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Vessel TRACK ID {track_id} - Motion Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Trajectory
    ax1 = axes[0, 0]
    trajectory = vessel_motion['trajectory']
    if trajectory:
        x_coords = [p[0] for p in trajectory]
        y_coords = [p[1] for p in trajectory]
        ax1.plot(x_coords, y_coords, 'b-', linewidth=2, label='Trajectory')
        ax1.scatter(x_coords[0], y_coords[0], color='green', s=100, marker='o', label='Start', zorder=5)
        ax1.scatter(x_coords[-1], y_coords[-1], color='red', s=100, marker='s', label='End', zorder=5)
        ax1.set_xlabel('X Position (pixels)')
        ax1.set_ylabel('Y Position (pixels)')
        ax1.set_title('Trajectory Path')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()  # Invert Y axis (image coordinates)
    
    # Plot 2: Speed over time
    ax2 = axes[0, 1]
    speeds = vessel_motion['speed_history']
    timestamps = vessel_motion['timestamps']
    if speeds and timestamps:
        ax2.plot(timestamps[1:], speeds, 'r-', linewidth=2, marker='o', markersize=3)
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Speed (px/s)')
        ax2.set_title('Speed Over Time')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Velocity components
    ax3 = axes[1, 0]
    velocities = vessel_motion['velocity_history']
    if velocities and timestamps:
        vx = [v[0] for v in velocities]
        vy = [v[1] for v in velocities]
        ax3.plot(timestamps[1:], vx, 'b-', linewidth=2, label='Vx', marker='o', markersize=3)
        ax3.plot(timestamps[1:], vy, 'r-', linewidth=2, label='Vy', marker='s', markersize=3)
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Velocity (px/s)')
        ax3.set_title('Velocity Components')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Acceleration magnitude
    ax4 = axes[1, 1]
    accelerations = vessel_motion['acceleration_history']
    if accelerations and timestamps:
        accel_magnitudes = [np.sqrt(a[0]**2 + a[1]**2) for a in accelerations]
        ax4.plot(timestamps[2:], accel_magnitudes, 'g-', linewidth=2, marker='o', markersize=3)
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Acceleration (px/s²)')
        ax4.set_title('Acceleration Magnitude')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Query and analyze specific vessels by TRACK ID")
    parser.add_argument("--results", type=str, required=True,
                       help="Path to tracking results JSON file")
    parser.add_argument("--track-id", type=int, default=None,
                       help="TRACK ID of vessel to query (if not provided, lists all vessels)")
    parser.add_argument("--list", action="store_true",
                       help="List all available vessels")
    parser.add_argument("--plot", action="store_true",
                       help="Generate plots for the specified vessel")
    parser.add_argument("--output-plot", type=str, default=None,
                       help="Path to save plot (default: show plot)")
    
    args = parser.parse_args()
    
    # Load tracking results
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return
    
    print(f"Loading tracking results from: {results_path}")
    tracking_data = load_tracking_results(results_path)
    
    # List all vessels if requested or no track_id specified
    if args.list or args.track_id is None:
        list_all_vessels(tracking_data)
    
    # Query specific vessel
    if args.track_id is not None:
        track_id = args.track_id
        vessel_motion = get_vessel_motion(tracking_data, track_id)
        
        if vessel_motion:
            print_vessel_details(vessel_motion)
            
            if args.plot:
                if args.output_plot:
                    plot_path = Path(args.output_plot)
                else:
                    plot_path = results_path.parent / f"vessel_{track_id}_motion_plot.png"
                plot_vessel_trajectory(tracking_data, track_id, plot_path)
        else:
            print(f"Error: Vessel TRACK ID {track_id} not found")
            list_all_vessels(tracking_data)


if __name__ == "__main__":
    main()
