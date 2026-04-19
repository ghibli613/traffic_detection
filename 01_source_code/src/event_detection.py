"""Event detection module for lane violations."""

import cv2
import numpy as np
from typing import List
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict


class EventType(Enum):
    """Types of events that can be detected."""
    TOUCH_LINE = "touch_line"
    CHANGE_LANE = "change_lane"
    TURN_SIGNAL = "turn_signal"


@dataclass
class Event:
    """Represents a detected event."""
    event_type: str
    vehicle_id: str
    start_frame: int
    end_frame: int
    start_time: str
    end_time: str
    signal_direction: str = None  # "left" or "right" for turn signals
    lane_from: int = None
    lane_to: int = None
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        result = {
            "event_type": self.event_type,
            "vehicle_id": self.vehicle_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }
        if self.signal_direction:
            result["signal"] = self.signal_direction
        if self.lane_from is not None:
            result["lane_from"] = self.lane_from
        if self.lane_to is not None:
            result["lane_to"] = self.lane_to
        return result


class EventDetector:
    """Detects lane violation events."""

    def __init__(self, config, fps: float = 24):
        self.config = config
        self.fps = fps
        self.events: List[Event] = []
        
        # Track vehicle states
        self.vehicle_lanes = {}  # vehicle_id -> current_lane_id
        self.vehicle_positions = defaultdict(list)  # vehicle_id -> list of (frame, center_x)
        self.vehicle_touch_events = defaultdict(list)  # vehicle_id -> list of touch events
        self.vehicle_lane_change_events = defaultdict(list)  # vehicle_id -> list of lane change events
        self.vehicle_turn_signal_events = defaultdict(list)  # vehicle_id -> list of turn signal events
        
        # Track event duration
        self.ongoing_events = {}  # event_key -> {start_frame, last_seen_frame, event_type, details}

    def detect_events(self, frame: int, vehicles: List, lane_lines: List, lanes: List, frame_image: np.ndarray) -> List[Event]:
        """
        Detect events in the current frame.
        
        Args:
            frame: Frame number
            vehicles: List of detected vehicles
            lane_lines: List of detected lane lines
            lanes: List of detected lanes
            frame_image: The frame image for analysis
            
        Returns:
            List of newly detected/completed events
        """
        new_events = []
        
        # Detect touch line events
        for vehicle in vehicles:
            # Update position tracking
            self.vehicle_positions[vehicle.vehicle_id].append((frame, vehicle.center_x))
            
            # Check for touching lane lines
            touch_events = self._detect_touch_line(vehicle, lane_lines, frame_image, frame)
            new_events.extend(touch_events)
            
            # Check for lane changes
            lane_change_events = self._detect_lane_change(vehicle, lanes, frame)
            new_events.extend(lane_change_events)
            
            # Check for turn signals
            turn_signal_events = self._detect_turn_signal(vehicle, frame_image, frame)
            new_events.extend(turn_signal_events)
        
        # Close events that are no longer observed in the current frame window.
        new_events.extend(self._close_stale_events(frame))
        self.events.extend(new_events)
        return new_events

    def _upsert_ongoing_event(self, event_key, frame_num: int, event_type: str, details):
        """Create a new ongoing event or refresh the last seen frame."""
        existing = self.ongoing_events.get(event_key)
        if existing is None:
            self.ongoing_events[event_key] = {
                "start_frame": frame_num,
                "last_seen_frame": frame_num,
                "event_type": event_type,
                "details": details,
            }
        else:
            existing["last_seen_frame"] = frame_num

    def _build_event(self, vehicle_id: str, event_type: str, start_frame: int, end_frame: int, details) -> Event:
        """Build a finalized Event object from tracked metadata."""
        event = Event(
            event_type=event_type,
            vehicle_id=vehicle_id,
            start_frame=start_frame,
            end_frame=end_frame,
            start_time=self._frame_to_time(start_frame),
            end_time=self._frame_to_time(end_frame),
        )

        if isinstance(details, dict):
            if "direction" in details:
                event.signal_direction = details["direction"]
            if "lane_from" in details:
                event.lane_from = details["lane_from"]
            if "lane_to" in details:
                event.lane_to = details["lane_to"]

        return event

    def _close_stale_events(self, current_frame: int) -> List[Event]:
        """Close ongoing events not seen for a short frame gap."""
        closed = []
        stale_keys = []
        max_gap_frames = max(2, self.config.event_detection.min_frames_for_event)

        for event_key, data in self.ongoing_events.items():
            if current_frame - data["last_seen_frame"] > max_gap_frames:
                duration = data["last_seen_frame"] - data["start_frame"] + 1
                if duration >= self.config.event_detection.min_frames_for_event:
                    vehicle_id = event_key[0]
                    closed.append(
                        self._build_event(
                            vehicle_id=vehicle_id,
                            event_type=data["event_type"],
                            start_frame=data["start_frame"],
                            end_frame=data["last_seen_frame"],
                            details=data["details"],
                        )
                    )
                stale_keys.append(event_key)

        for key in stale_keys:
            del self.ongoing_events[key]

        return closed

    def _detect_touch_line(self, vehicle, lane_lines: List, frame: np.ndarray, frame_num: int) -> List[Event]:
        """Detect if vehicle is touching a lane line."""
        events = []
        x1, y1, x2, y2 = vehicle.bbox
        center_x = vehicle.center_x

        h, w = frame.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return events
        
        # Check distance from center to lane lines
        for line in lane_lines:
            # Simple distance calculation from center to line
            line_x = np.mean(line.x_values)
            distance = abs(center_x - line_x)
            
            if distance < self.config.event_detection.touch_line_threshold:
                event_key = (vehicle.vehicle_id, "touch", int(line.line_id))
                
                self._upsert_ongoing_event(
                    event_key,
                    frame_num,
                    EventType.TOUCH_LINE.value,
                    {"line_id": int(line.line_id)},
                )
        
        return events

    def _detect_lane_change(self, vehicle, lanes: List, frame: int) -> List[Event]:
        """Detect if vehicle is changing lanes."""
        events = []
        center_x = vehicle.center_x
        
        # Determine current lane
        current_lane = None
        for lane in lanes:
            if lane.center_x - 50 < center_x < lane.center_x + 50:
                current_lane = lane.lane_id
                break
        
        if current_lane is None:
            return events
        
        # Track lane changes
        if vehicle.vehicle_id not in self.vehicle_lanes:
            self.vehicle_lanes[vehicle.vehicle_id] = current_lane
        else:
            prev_lane = self.vehicle_lanes[vehicle.vehicle_id]
            if prev_lane != current_lane:
                # Lane change detected
                event_key = (vehicle.vehicle_id, "lane_change", prev_lane, current_lane)
                
                self._upsert_ongoing_event(
                    event_key,
                    frame,
                    EventType.CHANGE_LANE.value,
                    {
                        "lane_from": prev_lane,
                        "lane_to": current_lane,
                    },
                )
                
                self.vehicle_lanes[vehicle.vehicle_id] = current_lane
        
        return events

    def _detect_turn_signal(self, vehicle, frame: np.ndarray, frame_num: int) -> List[Event]:
        """Detect turn signals (brake lights or indicators)."""
        events = []
        x1, y1, x2, y2 = vehicle.bbox
        
        h, w = frame.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return events

        # Extract vehicle region (focus on back for brake lights)
        # For simplicity, we'll check for bright red regions (brake lights)
        vehicle_region = frame[y1:y2, x1:x2]
        if vehicle_region.size == 0:
            return events
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(vehicle_region, cv2.COLOR_BGR2HSV)
        
        # Detect red (brake lights / indicators)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Detect yellow (turn signals)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        red_pixels = np.count_nonzero(mask_red)
        yellow_pixels = np.count_nonzero(mask_yellow)
        
        vehicle_area = (x2 - x1) * (y2 - y1)
        
        # If enough red pixels, it might be braking or indicator
        if red_pixels > vehicle_area * 0.02:  # 2% of vehicle area
            event_key = (vehicle.vehicle_id, "turn_signal", "right")
            
            self._upsert_ongoing_event(
                event_key,
                frame_num,
                EventType.TURN_SIGNAL.value,
                {
                    "direction": "right",
                },
            )
        
        if yellow_pixels > vehicle_area * 0.01:  # 1% of vehicle area
            event_key = (vehicle.vehicle_id, "turn_signal", "left")
            
            self._upsert_ongoing_event(
                event_key,
                frame_num,
                EventType.TURN_SIGNAL.value,
                {
                    "direction": "left",
                },
            )
        
        return events

    def finalize_events(self, current_frame: int) -> List[Event]:
        """Finalize ongoing events."""
        finalized = []

        for event_key, data in self.ongoing_events.items():
            duration = data["last_seen_frame"] - data["start_frame"] + 1
            if duration >= self.config.event_detection.min_frames_for_event:
                vehicle_id = event_key[0]
                finalized.append(
                    self._build_event(
                        vehicle_id=vehicle_id,
                        event_type=data["event_type"],
                        start_frame=data["start_frame"],
                        end_frame=data["last_seen_frame"],
                        details=data["details"],
                    )
                )
        
        self.ongoing_events.clear()
        self.events.extend(finalized)
        return finalized

    def get_active_events(self, current_frame: int) -> List[Event]:
        """Get synthetic active events for in-frame overlay rendering."""
        active = []
        for event_key, data in self.ongoing_events.items():
            vehicle_id = event_key[0]
            active.append(
                self._build_event(
                    vehicle_id=vehicle_id,
                    event_type=data["event_type"],
                    start_frame=data["start_frame"],
                    end_frame=current_frame,
                    details=data["details"],
                )
            )
        return active

    def _frame_to_time(self, frame_num: int) -> str:
        """Convert frame number to HH:MM:SS format."""
        seconds = frame_num / self.fps
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        hours = minutes // 60
        minutes = minutes % 60
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def get_all_events(self) -> List[Event]:
        """Get all detected events."""
        return self.events
