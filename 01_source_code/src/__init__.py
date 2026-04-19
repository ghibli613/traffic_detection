"""Lane Violation Detection System - Main Package."""

from .config import get_default_config, SystemConfig
from .lane_detection import LaneDetector, Lane, LaneLine
from .vehicle_detection import VehicleDetector, Vehicle
from .event_detection import EventDetector, Event, EventType
from .pipeline import VideoProcessor

__version__ = "1.0.0"
__all__ = [
    "get_default_config",
    "SystemConfig",
    "LaneDetector",
    "Lane",
    "LaneLine",
    "VehicleDetector",
    "Vehicle",
    "EventDetector",
    "Event",
    "EventType",
    "VideoProcessor",
]
