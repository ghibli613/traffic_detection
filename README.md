# Traffic Detection

An AI vision pipeline that processes traffic video and produces lane-discipline events, an annotated output video, and structured JSON data.

## Project Structure

```
traffic_detection/
├── 01_source_code/
│   ├── src/
│   │   ├── config.py               # Configuration dataclasses
│   │   ├── lane_detection.py       # Canny + Hough lane detection
│   │   ├── vehicle_detection.py    # YOLO 11n vehicle detection & tracking
│   │   ├── event_detection.py      # Rule-based event lifecycle engine
│   │   └── pipeline.py             # Orchestrator — reads video, writes outputs
│   └── ui/
│       └── app.py                  # Streamlit web interface
├── 02_report/
│   ├── formal_report.md
│   └── formal_report.pdf
├── 03_video_demo/                  # Output: annotated event video
├── 04_json_event_output/           # Output: JSON event log
├── 05_lane_line_detection_output/  # Output: lane detection image
├── colab_test_pipeline.ipynb       # Colab validation notebook
├── requirements.txt                # Pinned dependencies (Colab target)
└── README.md
```

## Environment

| | Version |
|---|---|
| Python | 3.12.13 |
| torch | 2.10.0+cu128 |
| torchvision | 0.25.0+cu128 |
| numpy | 2.0.2 |
| opencv-python-headless | 4.13.0.92 |
| ultralytics | ≥ 8.3.0 |
| CUDA | 12.8 |

## Usage

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Web UI

```bash
cd 01_source_code/ui
streamlit run app.py
```

Open `http://localhost:8501`, upload a traffic video, and click **Analyze**.

### CLI

Run the pipeline directly from the repository root:

```bash
python 01_source_code/src/pipeline.py /path/to/your/video.mp4 analysis
```

Or run it from inside the source folder:

```bash
cd 01_source_code/src
python pipeline.py /path/to/your/video.mp4 analysis
```

Arguments:

- `video_path`: input traffic video file
- `output_prefix`: optional prefix for generated outputs; `analysis` is used above as an example

This generates:

- `03_video_demo/<prefix>_events.mp4`
- `04_json_event_output/<prefix>_events.json`
- `05_lane_line_detection_output/<prefix>_lanes.png`


### Python API

```python
from pathlib import Path
import sys
sys.path.insert(0, "01_source_code/src")

from config import get_default_config
from pipeline import VideoProcessor

config = get_default_config()
vp = VideoProcessor(
    config=config,
    video_dir="03_video_demo",
    json_dir="04_json_event_output",
    lane_dir="05_lane_line_detection_output",
)
vp.process_video("test.mp4", output_prefix="analysis")
```

## Outputs

| File | Description |
|------|-------------|
| `03_video_demo/<prefix>_events.mp4` | Annotated video with bounding boxes and event overlays |
| `04_json_event_output/<prefix>_events.json` | Structured JSON event log |
| `05_lane_line_detection_output/<prefix>_lanes.png` | Lane line detection image |

### JSON Schema

```json
{
  "events": [
    {
      "event_type": "touch_line",
      "vehicle_id": "VE586",
      "start_time": "00:06",
      "end_time": "00:07"
    },
    {
      "event_type": "change_lane",
      "vehicle_id": "VE586",
      "start_time": "00:08",
      "end_time": "00:11",
      "lane_from": 2,
      "lane_to": 3
    },
    {
      "event_type": "turn_signal",
      "vehicle_id": "97963",
      "signal": "right",
      "start_time": "00:25",
      "end_time": "00:40"
    }
  ]
}
```

## Detected Events

| Event | Trigger |
|-------|---------|
| `touch_line` | Vehicle bounding box edge within threshold pixels of a lane line |
| `change_lane` | Vehicle centroid crosses from one lane region to another |
| `turn_signal` | Red/yellow pixel ratio in vehicle region exceeds threshold |

## Configuration

Edit `01_source_code/src/config.py` to tune parameters:

```python
lane_detection.canny_threshold1 = 50
lane_detection.canny_threshold2 = 150
vehicle_detection.confidence_threshold = 0.5
event_detection.touch_line_threshold = 10.0   # pixels
event_detection.lane_change_threshold = 0.3   # lane-width fraction
video.frame_skip = 1                          # process every Nth frame
```

## Model

Detection uses **YOLO 11n** (`yolo11n.pt`). The model file is downloaded automatically by `ultralytics` on first run and is excluded from the repository via `.gitignore`.

## System Requirements

- Python 3.8+
- CPU: 4+ cores, 4 GB RAM minimum (8 GB recommended)
- GPU: NVIDIA with 2+ GB VRAM (optional, improves throughput)

## API Reference

### VideoProcessor

```python
from src.pipeline import VideoProcessor
from src.config import get_default_config

config = get_default_config()
processor = VideoProcessor(
    config,
    video_dir="03_video_demo",
    json_dir="04_json_event_output",
    lane_dir="05_lane_line_detection_output",
)

# Process a video
output_video, events_json, lane_vis = processor.process_video(
    "traffic.mp4",
    output_prefix="analysis"
)
```

### Custom Configuration

```python
from src.config import SystemConfig, LaneDetectionConfig, VehicleDetectionConfig

config = SystemConfig(
    lane_detection=LaneDetectionConfig(
        canny_threshold1=50,
        canny_threshold2=150,
    ),
    vehicle_detection=VehicleDetectionConfig(
        model_name="yolo11n",
        confidence_threshold=0.4,
        device="cuda",
    ),
)
```