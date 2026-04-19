"""Lane detection module using image processing techniques."""

import cv2
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class Lane:
    """Represents a detected lane."""
    lane_id: int
    left_line_id: int
    right_line_id: int
    center_x: float


@dataclass
class LaneLine:
    """Represents a detected lane line."""
    line_id: int
    x_values: np.ndarray
    y_values: np.ndarray
    slope: float
    intercept: float
    orientation: str  # "vertical" or "diagonal"


class LaneDetector:
    """Detects lane boundaries and lines from video frames."""

    def __init__(self, config):
        self.config = config
        self.line_counter = 0
        self.previous_lines = []

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for lane detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
        
        return morph

    def _detect_edges(self, processed: np.ndarray) -> np.ndarray:
        """Detect edges using Canny edge detection."""
        edges = cv2.Canny(
            processed,
            self.config.lane_detection.canny_threshold1,
            self.config.lane_detection.canny_threshold2,
        )
        return edges

    def _detect_hough_lines(self, edges: np.ndarray, frame_height: int, frame_width: int) -> List[Tuple[int, int, int, int]]:
        """Detect lines using Hough line transform."""
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.config.lane_detection.hough_lines_threshold,
            minLineLength=self.config.lane_detection.hough_lines_min_line_length,
            maxLineGap=self.config.lane_detection.hough_lines_max_line_gap,
        )
        
        detected_lines = []
        if lines is not None:
            # Filter lines by angle (remove horizontal lines)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x1 != x2:  # Avoid division by zero
                    angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    # Keep nearly vertical lines (lane lines should be vertical-ish)
                    if 60 <= angle <= 120:
                        detected_lines.append((x1, y1, x2, y2))
        
        return detected_lines

    def _cluster_and_label_lines(self, lines: List[Tuple[int, int, int, int]]) -> List[LaneLine]:
        """Cluster similar lines and label them with unique IDs."""
        if not lines:
            return []

        # Sort lines by x-coordinate of their center
        sorted_lines = sorted(lines, key=lambda l: (l[0] + l[2]) / 2)
        
        labeled_lines = []
        used = set()
        
        for idx, line in enumerate(sorted_lines):
            if idx in used:
                continue
            
            x1, y1, x2, y2 = line
            
            # Calculate line properties
            if x1 != x2:
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
            else:
                slope = float('inf')
                intercept = x1
            
            orientation = "vertical" if abs(slope) > 2 else "diagonal"
            
            self.line_counter += 1
            lane_line = LaneLine(
                line_id=self.line_counter,
                x_values=np.array([x1, x2]),
                y_values=np.array([y1, y2]),
                slope=slope,
                intercept=intercept,
                orientation=orientation,
            )
            
            labeled_lines.append(lane_line)
            used.add(idx)
        
        return labeled_lines

    def _group_lines_into_lanes(self, lines: List[LaneLine]) -> List[Lane]:
        """Group lane lines into lanes."""
        if len(lines) < 2:
            return []
        
        lanes = []
        lane_id = 0
        
        # Sort lines by x-coordinate
        sorted_lines = sorted(lines, key=lambda l: np.mean(l.x_values))
        
        # Pair consecutive lines to form lanes
        for i in range(len(sorted_lines) - 1):
            lane_id += 1
            left_line = sorted_lines[i]
            right_line = sorted_lines[i + 1]
            
            center_x = (np.mean(left_line.x_values) + np.mean(right_line.x_values)) / 2
            
            lanes.append(Lane(
                lane_id=lane_id,
                left_line_id=left_line.line_id,
                right_line_id=right_line.line_id,
                center_x=center_x,
            ))
        
        return lanes

    def detect(self, frame: np.ndarray) -> Tuple[List[LaneLine], List[Lane], np.ndarray]:
        """
        Detect lane lines and lanes in a frame.
        
        Returns:
            Tuple of (lane_lines, lanes, visualization)
        """
        # Preprocess frame
        processed = self._preprocess_frame(frame)
        
        # Detect edges
        edges = self._detect_edges(processed)
        
        # Detect lines
        lines = self._detect_hough_lines(edges, frame.shape[0], frame.shape[1])
        
        # Cluster and label lines
        lane_lines = self._cluster_and_label_lines(lines)
        
        # Group lines into lanes
        lanes = self._group_lines_into_lanes(lane_lines)
        
        # Create visualization
        vis = self._create_visualization(frame, lane_lines, lanes)
        
        return lane_lines, lanes, vis

    def _create_visualization(self, frame: np.ndarray, lines: List[LaneLine], lanes: List[Lane]) -> np.ndarray:
        """Create visualization of detected lanes and lines."""
        vis = frame.copy()
        
        # Draw lane lines
        colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
        for line in lines:
            x1, y1 = int(line.x_values[0]), int(line.y_values[0])
            x2, y2 = int(line.x_values[1]), int(line.y_values[1])
            color = colors[line.line_id % len(colors)]
            cv2.line(vis, (x1, y1), (x2, y2), color, 3)
            cv2.putText(vis, f"L{line.line_id}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw lane labels
        for lane in lanes:
            x = int(lane.center_x)
            y = 50 + lane.lane_id * 30
            cv2.putText(vis, f"Lane {lane.lane_id}", (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return vis
