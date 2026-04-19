"""Vehicle detection module using Ultralytics YOLO."""

import cv2
import numpy as np
import torch
from typing import List, Tuple
from dataclasses import dataclass, field
from ultralytics import YOLO


@dataclass
class Vehicle:
    """Represents a detected vehicle."""
    vehicle_id: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center_x: float
    center_y: float
    confidence: float
    class_name: str
    color: Tuple[int, int, int] = field(default_factory=lambda: (0, 255, 0))


class VehicleDetector:
    """Detects vehicles in video frames using Ultralytics YOLO."""

    def __init__(self, config):
        self.config = config
        self.vehicle_id_map = {}  # Map from detection to persistent ID
        self.next_id = 1
        self.load_model()

    def load_model(self):
        """Load Ultralytics YOLO model and choose inference device."""
        try:
            requested_device = self.config.vehicle_detection.device
            if requested_device == "cuda" and not torch.cuda.is_available():
                requested_device = "cpu"

            # Accept both explicit checkpoint names and legacy names from older config.
            model_name = self.config.vehicle_detection.model_name
            if model_name == "yolov5s":
                model_name = "yolo11n.pt"

            self.infer_device = requested_device
            self.model = YOLO(model_name)
        except Exception as e:
            print(f"Error loading Ultralytics YOLO: {e}. Using fallback detection.")
            self.model = None

    def detect(self, frame: np.ndarray) -> List[Vehicle]:
        """
        Detect vehicles in a frame.
        
        Returns:
            List of detected vehicles
        """
        vehicles = []
        
        if self.model is None:
            return vehicles
        
        try:
            # Run inference
            results = self.model.predict(
                source=frame,
                imgsz=640,
                conf=self.config.vehicle_detection.confidence_threshold,
                iou=self.config.vehicle_detection.iou_threshold,
                device=self.infer_device,
                verbose=False,
            )
            if not results:
                return vehicles

            result = results[0]
            predictions = result.boxes
            if predictions is None:
                return vehicles

            names = result.names
            
            # Process detections
            for pred in predictions:
                x1, y1, x2, y2 = pred.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = float(pred.conf[0].item())
                cls = int(pred.cls[0].item())
                
                # Filter for vehicle classes (car, truck, bus, motorcycle)
                class_name = names[cls] if isinstance(names, dict) else names[cls]
                if class_name not in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']:
                    continue
                
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Simple tracking: assign persistent IDs based on proximity
                vehicle_id = self._get_or_create_vehicle_id((x1, y1, x2, y2))
                
                vehicle = Vehicle(
                    vehicle_id=vehicle_id,
                    bbox=(x1, y1, x2, y2),
                    center_x=center_x,
                    center_y=center_y,
                    confidence=conf,
                    class_name=class_name,
                    color=self._get_color_for_id(vehicle_id),
                )
                
                vehicles.append(vehicle)
        
        except Exception as e:
            print(f"Error during detection: {e}")
        
        return vehicles

    def _get_or_create_vehicle_id(self, bbox: Tuple[int, int, int, int]) -> str:
        """Get or create a persistent ID for a vehicle."""
        # Simple proximity-based tracking
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        # Find closest previous detection
        closest_id = None
        min_distance = 50  # pixels
        
        for (prev_x1, prev_y1, prev_x2, prev_y2), vid in self.vehicle_id_map.items():
            prev_center = ((prev_x1 + prev_x2) / 2, (prev_y1 + prev_y2) / 2)
            distance = np.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_id = vid
        
        if closest_id:
            del self.vehicle_id_map[list(self.vehicle_id_map.keys())[
                list(self.vehicle_id_map.values()).index(closest_id)
            ]]
            self.vehicle_id_map[bbox] = closest_id
            return closest_id
        
        # Create new ID
        new_id = f"V{self.next_id:03d}"
        self.next_id += 1
        self.vehicle_id_map[bbox] = new_id
        
        return new_id

    def _get_color_for_id(self, vehicle_id: str) -> Tuple[int, int, int]:
        """Get a consistent color for a vehicle ID."""
        hash_val = hash(vehicle_id) % 6
        colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        return colors[hash_val]

    def draw_detections(self, frame: np.ndarray, vehicles: List[Vehicle]) -> np.ndarray:
        """Draw vehicle detections on frame."""
        result = frame.copy()
        
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle.bbox
            
            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), vehicle.color, 2)
            
            # Draw vehicle ID and class
            label = f"{vehicle.vehicle_id} ({vehicle.class_name})"
            cv2.putText(result, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, vehicle.color, 2)
            
            # Draw center point
            cv2.circle(result, (int(vehicle.center_x), int(vehicle.center_y)), 3, vehicle.color, -1)
        
        return result
