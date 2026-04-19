"""Main video processing pipeline."""

import cv2
import json
from typing import Tuple, List
from pathlib import Path
import numpy as np

from lane_detection import LaneDetector
from vehicle_detection import VehicleDetector
from event_detection import EventDetector, Event
from config import get_default_config


class VideoProcessor:
    """Main pipeline for processing traffic videos."""

    def __init__(
        self,
        config=None,
        video_dir: str = "03_video_demo",
        json_dir: str = "04_json_event_output",
        lane_dir: str = "05_lane_line_detection_output",
    ):
        self.config = config or get_default_config()
        self.video_dir = Path(video_dir)
        self.json_dir = Path(json_dir)
        self.lane_dir = Path(lane_dir)
        for d in (self.video_dir, self.json_dir, self.lane_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Initialize detectors
        self.lane_detector = LaneDetector(self.config)
        self.vehicle_detector = VehicleDetector(self.config)
        self.event_detector = None  # Will be initialized when we know FPS

        # Results storage
        self.events: List[Event] = []
        self.lane_visualizations: List[np.ndarray] = []

    def process_video(self, video_path: str, output_prefix: str = "output") -> Tuple[str, str, str]:
        """
        Process a video and generate outputs.
        
        Args:
            video_path: Path to input video
            output_prefix: Prefix for output files
            
        Returns:
            Tuple of (output_video_path, events_json_path, lane_visualization_path)
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize event detector with FPS
        self.event_detector = EventDetector(self.config, fps=fps)
        
        # Setup output video writer
        output_video_path = self.video_dir / f"{output_prefix}_events.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))
        
        frame_count = 0
        processed_count = 0
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}, Resolution: {frame_width}x{frame_height}, Total frames: {total_frames}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if configured
            if frame_count % self.config.video.frame_skip != 0:
                frame_count += 1
                continue
            
            # Check max frames
            if self.config.max_frames and processed_count >= self.config.max_frames:
                break
            
            # Detect lanes
            lane_lines, lanes, lane_vis = self.lane_detector.detect(frame)
            
            # Detect vehicles
            vehicles = self.vehicle_detector.detect(frame)
            
            # Draw vehicles on frame
            vis_frame = self.vehicle_detector.draw_detections(frame, vehicles)
            
            # Detect/track events for current frame.
            self.event_detector.detect_events(frame_count, vehicles, lane_lines, lanes, frame)
            
            # Draw currently active events so overlays appear during the event window.
            vis_frame = self._draw_events(vis_frame, self.event_detector.get_active_events(frame_count), frame_count)
            
            # Add lane visualization
            if len(self.lane_visualizations) == 0 and len(lanes) > 0:
                self.lane_visualizations.append(lane_vis)
            
            # Write frame to output video
            out.write(vis_frame)
            
            processed_count += 1
            if processed_count % 30 == 0:
                print(f"Processed {processed_count} frames...")
            
            frame_count += 1
        
        # Finalize any ongoing events
        self.event_detector.finalize_events(frame_count)
        self.events = self.event_detector.get_all_events()
        
        cap.release()
        out.release()
        
        print(f"Finished processing. Total events detected: {len(self.events)}")
        
        # Save outputs
        events_json_path = self._save_events_json(output_prefix)
        lane_vis_path = self._save_lane_visualization(output_prefix)
        
        return str(output_video_path), events_json_path, lane_vis_path

    def _draw_events(self, frame: np.ndarray, events: List[Event], current_frame: int) -> np.ndarray:
        """Draw detected events on the frame."""
        result = frame.copy()
        
        # Draw active events (events that include current frame)
        for event in events:
            if event.start_frame <= current_frame <= event.end_frame:
                # Draw event text
                y_offset = 100 + events.index(event) * 30
                text = f"{event.vehicle_id}: {event.event_type} ({event.start_time}-{event.end_time})"
                
                if event.event_type == "turn_signal":
                    text += f" [{event.signal_direction}]"
                
                cv2.putText(result, text, (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return result

    def _save_events_json(self, prefix: str) -> str:
        """Save detected events to JSON file."""
        output_path = self.json_dir / f"{prefix}_events.json"
        
        events_data = {
            "events": [event.to_dict() for event in self.events],
            "total_events": len(self.events),
        }
        
        with open(output_path, 'w') as f:
            json.dump(events_data, f, indent=2)
        
        print(f"Saved events to: {output_path}")
        return str(output_path)

    def _save_lane_visualization(self, prefix: str) -> str:
        """Save lane visualization as image."""
        if not self.lane_visualizations:
            return None

        output_path = self.lane_dir / f"{prefix}_lanes.png"
        cv2.imwrite(str(output_path), self.lane_visualizations[0])
        
        print(f"Saved lane visualization to: {output_path}")
        return str(output_path)


def main():
    """Example usage of the video processor."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <video_path> [output_prefix]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else "output"
    
    config = get_default_config()
    processor = VideoProcessor(config)  # outputs go to 03_video_demo/, 04_json_event_output/, 05_lane_line_detection_output/

    try:
        output_video, events_json, lane_vis = processor.process_video(video_path, output_prefix)
        print(f"\nResults:")
        print(f"  Output video: {output_video}")
        print(f"  Events JSON: {events_json}")
        print(f"  Lane visualization: {lane_vis}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
