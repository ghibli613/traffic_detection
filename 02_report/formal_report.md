# AI Vision Engineer Task Report: Lane Violation Event Detection

## 1. Project Summary
This report documents the design and implementation of an AI Vision system that analyzes traffic video and outputs lane-discipline related events. The system focuses on visual evidence extraction rather than legal judgment. It detects lane lines, infers lane regions, tracks vehicles, and exports event records for review.

## 2. Problem Scope and Objectives
### 2.1 Objective
Build an AI Vision pipeline that detects events relevant to lane-discipline analysis:
- Lane and line detection with IDs
- Vehicle touching lane line
- Vehicle lane change / lane crossing
- Vehicle turn signal request (left/right)
- Export all events as JSON with timestamps

### 2.2 Non-Objective
The system does not make legal conclusions. It provides machine-generated visual and temporal evidence for later human/legal interpretation.

## 3. Architecture Diagram
```text
+----------------------------+
| Input Traffic Video        |
+-------------+--------------+
              |
              v
+----------------------------+
| Frame Loader / Preprocess  |
| - Decode frames            |
| - Resize/normalize         |
+-------------+--------------+
              |
     +--------+--------+
     |                 |
     v                 v
+----------+      +----------------+
| Lane     |      | Vehicle        |
| Detector |      | Detector       |
| (CV)     |      | (YOLOv5)       |
+----+-----+      +--------+-------+
     |                     |
     +----------+----------+
                |
                v
+----------------------------+
| Event Detector             |
| - touch_line               |
| - change_lane              |
| - turn_signal              |
+-------------+--------------+
              |
     +--------+--------+----------------+
     |                 |                |
     v                 v                v
+-----------+    +--------------+   +----------------+
| Lane Image |    | Event JSON   |   | Annotated Video |
+-----------+    +--------------+   +----------------+
```

## 4. Flow Diagram
```text
START
  |
  v
Load Video -> Read FPS/Size -> Initialize Modules
  |
  v
For each frame:
  1) Detect lane lines
  2) Label line IDs and infer lane IDs
  3) Detect vehicles and track IDs
  4) Evaluate event rules:
     - touch_line
     - change_lane
     - turn_signal
  5) Draw overlays
  6) Append active events
  |
  v
Finalize ongoing events
  |
  v
Export:
  - lane_line_detection_output.png
  - events_output.json
  - video_demo.mp4
  |
  v
END
```

## 5. Overall Approach
The implementation uses a modular hybrid approach:
- Classical computer vision for lane/line extraction (edge + line transform)
- Deep learning detector for vehicle localization (YOLOv5)
- Rule-based temporal logic for event generation

Why hybrid:
- Lane geometry is efficiently captured by deterministic CV methods.
- Vehicle detection quality is best handled by pretrained modern detectors.
- Event rules remain transparent and auditable when expressed explicitly.

## 6. Key Design Decisions
1. Modular pipeline
- Lane, vehicle, and event logic are split into separate modules for maintainability and replacement.

2. Explainability-first event logic
- Event rules are simple and inspectable: proximity to line, lane index transitions, signal region cues.

3. Submission-focused outputs
- Pipeline always emits three core artifacts: image, video, JSON.

4. Cost-conscious model strategy
- Local/self-hosted model path chosen to avoid per-frame API charges.

5. Practical limitation handling
- System captures events that support review, instead of claiming legal certainty.

## 7. Implementation Overview
### 7.1 Source modules
- `src/lane_detection.py`: lane line detection and lane inference
- `src/vehicle_detection.py`: vehicle detection and ID continuity
- `src/event_detection.py`: temporal event extraction
- `src/pipeline.py`: orchestrates end-to-end processing
- `src/config.py`: parameter configuration
- `ui/app.py`: Streamlit UI for upload/analyze/review

### 7.2 Event schema
The output JSON uses this structure:
```json
{
  "events": [
    {
      "event_type": "touch_line",
      "vehicle_id": "VE586",
      "start_time": "00:06",
      "end_time": "00:07"
    }
  ]
}
```

### 7.3 Output artifacts
- `lane_line_detection_output.png` — visualized lane/line IDs
- `events_output.json` — structured event data with timestamps
- `video_demo.mp4` — annotated output video with event overlays

## 8. Results
The current submission includes a generated demonstration run with required artifacts.

### 8.1 Functional results
- Lane line IDs and lane IDs are visualized in the lane output image.
- Event timeline is rendered in demo video with highlighted vehicles.
- JSON export contains event list and timing windows for each event type.

### 8.2 Coverage against task requirements
- Lane and line detection: done
- Vehicle touch line event: done
- Lane change event: done
- Turn signal request event: done
- JSON export: done
- UI flow available: done (upload -> run -> outputs)

## 9. Limitations
1. Camera/viewpoint dependence
- Performance decreases for severe camera tilt or occlusion.

2. Lane quality sensitivity
- Faded or obstructed lane markings reduce line detection reliability.

3. Turn signal ambiguity
- Indicator detection can be affected by reflections, brake lights, and compression artifacts.

4. Rule-based thresholds
- Fixed thresholds may require tuning across road types and weather conditions.

5. Tracking complexity
- Dense traffic and long occlusions can cause temporary ID switches.

## 10. Evaluation Method
### 10.1 Proposed offline evaluation protocol
1. Build a labeled validation set with frame ranges for:
- touch_line
- change_lane
- turn_signal

2. Measure:
- Event precision
- Event recall
- Event F1 score
- Timestamp overlap IoU for event windows

3. Lane quality metrics:
- Line detection precision/recall against manually annotated lane lines
- Lane assignment consistency over time

### 10.2 Practical review process
- Compare generated video overlays against source clip.
- Cross-check JSON event time windows with manual spot checks.
- Record false positives/false negatives per category.

## 11. Session Resource Usage (CPU / GPU)
Measured environment for this implementation session:
- CPU: 13th Gen Intel Core i7-1370P
- Logical CPUs: 20
- GPU: No NVIDIA GPU detected in current environment

Observed execution mode in this environment:
- CPU-only processing
- Artifact generation and demo rendering completed successfully

Expected resource profile (typical):
- CPU-only: moderate processing speed, lower setup complexity
- GPU-accelerated: higher throughput, better for batch scale

## 12. Vision API Usage and Cost
This submission uses self-hosted/local processing for core pipeline logic.

- Vision API token usage: 0
- External vision API cost: 0

If a cloud Vision API is used in a future variant, cost should be estimated as:
- cost_per_frame x analyzed_frames_per_video x number_of_videos

## 13. Self-Hosted Model Performance Characteristics and Trade-offs
### 13.1 Advantages
- Zero per-frame API cost
- Better data control and privacy
- Offline-capable deployment
- Stable behavior without network latency

### 13.2 Trade-offs
- Requires local compute resources
- Model optimization/tuning burden is on developer
- Throughput may be limited on CPU-only machines
- Maintenance includes dependency and model version management

## 14. Plan to Improve Output Quality
1. Lane detection robustness
- Add temporal smoothing and lane hypothesis tracking across frames.

2. Tracking quality
- Upgrade to stronger multi-object tracking for fewer ID switches.

3. Turn signal detection
- Train dedicated classifier on rear-light/indicator crops with temporal blink modeling.

4. Domain adaptation
- Tune thresholds by scene profile (day/night/rain/highway/urban).

5. Confidence-driven postprocessing
- Add event confidence scores and suppress low-confidence event bursts.

## 15. Scalability Considerations
As number of videos, scenes, or conditions grows:

1. Pipeline parallelization
- Process videos in worker queues.
- Batch frame inference where possible.

2. Config profiles by scenario
- Maintain preset configs for highway, city, night, and rain.

3. Storage and indexing
- Store JSON events in searchable datastore for rapid retrieval and analytics.

4. Monitoring and drift detection
- Track event-rate distributions by camera and time.
- Trigger retraining/tuning when drift is detected.

5. Deployment strategy
- CPU edge mode for low-cost distributed sites.
- GPU centralized mode for high-throughput processing centers.

## 16. Reproducibility
To reproduce all outputs from source:

```bash
pip install -r requirements.txt
cd src
python pipeline.py <input_video.mp4> [output_prefix]
```

Or use the Streamlit interface:

```bash
streamlit run ui/app.py
```

## 17. Conclusion
The system successfully demonstrates a working lane-discipline analysis pipeline using a hybrid of classical CV and deep learning. The modular design keeps each detection stage independently verifiable and replaceable. Key future work focuses on temporal consistency in lane tracking, stronger vehicle tracking under occlusion, and dedicated indicator detection to replace the current color heuristic.
