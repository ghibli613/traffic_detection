"""Configuration for the lane violation detection system."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class LaneDetectionConfig:
    """Configuration for lane detection."""
    roi_vertices: Tuple[Tuple[int, int], ...] = None  # Region of interest
    canny_threshold1: int = 50
    canny_threshold2: int = 150
    hough_lines_threshold: int = 50
    hough_lines_min_line_length: int = 50
    hough_lines_max_line_gap: int = 10
    line_thickness: int = 2


@dataclass
class VehicleDetectionConfig:
    """Configuration for vehicle detection."""
    model_name: str = "yolo11n.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    device: str = "cpu"  # "cuda" or "cpu"


@dataclass
class EventDetectionConfig:
    """Configuration for event detection."""
    touch_line_threshold: float = 10.0  # pixels
    lane_change_threshold: float = 0.3  # fraction of vehicle width
    turn_signal_brightness_threshold: int = 150
    min_frames_for_event: int = 5


@dataclass
class VideoConfig:
    """Configuration for video processing."""
    target_fps: int = 24
    frame_skip: int = 1
    output_video_fps: int = 24
    output_video_format: str = "mp4v"
    output_video_codec: str = "mp4v"


@dataclass
class SystemConfig:
    """Overall system configuration."""
    lane_detection: LaneDetectionConfig
    vehicle_detection: VehicleDetectionConfig
    event_detection: EventDetectionConfig
    video: VideoConfig
    debug: bool = False
    max_frames: int = None  # Process all frames if None


def get_default_config() -> SystemConfig:
    """Get default system configuration."""
    return SystemConfig(
        lane_detection=LaneDetectionConfig(),
        vehicle_detection=VehicleDetectionConfig(),
        event_detection=EventDetectionConfig(),
        video=VideoConfig(),
    )
