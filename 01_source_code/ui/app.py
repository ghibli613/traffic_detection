"""Streamlit web UI for the lane violation detection system."""

import streamlit as st
import cv2
import json
from pathlib import Path
import numpy as np
from typing import Tuple
import tempfile
import os

# Add src to path
import sys
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from pipeline import VideoProcessor
from config import get_default_config


def create_ui():
    """Create the Streamlit UI."""
    st.set_page_config(
        page_title="Lane Violation Detection",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.title("🚗 Lane Violation Detection System")
    st.markdown("### AI Vision System for Traffic Analysis")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        max_frames = st.number_input(
            "Max frames to process",
            min_value=1,
            max_value=10000,
            value=500,
            help="Limit processing to N frames (lower = faster)"
        )
        
        device = st.selectbox(
            "Processing Device",
            ["cpu", "cuda"],
            help="Use GPU if available"
        )
        
        debug_mode = st.checkbox("Debug Mode", value=False)
    
    # Main content
    tabs = st.tabs(["Home", "Process Video", "Results", "About"])
    
    with tabs[0]:  # Home tab
        st.markdown("""
        ## Welcome to Lane Violation Detection System
        
        This system analyzes traffic videos to detect:
        - 🛣️ **Lane Detection** - Identify and label lane boundaries
        - 🚙 **Vehicle Tracking** - Detect and track vehicles
        - ⚠️ **Lane Violations** - Detect vehicles touching, changing, or violating lanes
        - 🚦 **Turn Signals** - Identify turn signal usage
        
        ### Quick Start:
        1. Go to **Process Video** tab
        2. Upload a traffic video
        3. Click **Analyze** to process
        4. View results in **Results** tab
        """)
        
        st.info("💡 Tip: Start with a 10-30 second video for quick testing")
    
    with tabs[1]:  # Process Video tab
        st.header("Upload and Process Video")
        
        # File upload
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
        
        if uploaded_file is not None:
            # Create temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_video_path = tmp_file.name
            
            st.success(f"✓ File uploaded: {uploaded_file.name}")
            
            # Show video info
            cap = cv2.VideoCapture(temp_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Duration", f"{duration:.1f}s")
            col2.metric("FPS", f"{fps:.1f}")
            col3.metric("Total Frames", frame_count)
            
            # Process button
            if st.button("▶️ Analyze Video", type="primary", use_container_width=True):
                with st.spinner("Processing video... This may take a while"):
                    try:
                        # Configure
                        config = get_default_config()
                        config.max_frames = max_frames
                        config.vehicle_detection.device = device
                        config.debug = debug_mode
                        
                        # Process — each artifact type goes to its own output folder
                        base = Path(__file__).resolve().parent.parent.parent
                        processor = VideoProcessor(
                            config,
                            video_dir=str(base / "03_video_demo"),
                            json_dir=str(base / "04_json_event_output"),
                            lane_dir=str(base / "05_lane_line_detection_output"),
                        )
                        output_video, events_json, lane_vis = processor.process_video(
                            temp_video_path,
                            output_prefix="analysis"
                        )
                        
                        # Store results in session
                        st.session_state.output_video = output_video
                        st.session_state.events_json = events_json
                        st.session_state.lane_vis = lane_vis
                        st.session_state.events = processor.events
                        
                        st.success("✓ Analysis complete!")
                        st.info("Go to **Results** tab to view outputs")
                        
                    except Exception as e:
                        st.error(f"Error processing video: {e}")
                        import traceback
                        st.error(traceback.format_exc())
                    finally:
                        # Cleanup
                        try:
                            os.unlink(temp_video_path)
                        except:
                            pass
    
    with tabs[2]:  # Results tab
        if "output_video" not in st.session_state:
            st.info("No results yet. Process a video first.")
        else:
            st.header("Analysis Results")
            
            # Results tabs
            result_tabs = st.tabs(["Video", "Events", "Lane Detection"])
            
            with result_tabs[0]:  # Video tab
                st.subheader("Processed Video with Detected Events")
                try:
                    st.video(st.session_state.output_video)
                except Exception as e:
                    st.error(f"Error displaying video: {e}")
            
            with result_tabs[1]:  # Events tab
                st.subheader("Detected Events")
                
                if st.session_state.events:
                    # Events statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    touch_events = len([e for e in st.session_state.events if e.event_type == "touch_line"])
                    change_events = len([e for e in st.session_state.events if e.event_type == "change_lane"])
                    signal_events = len([e for e in st.session_state.events if e.event_type == "turn_signal"])
                    
                    col1.metric("Total Events", len(st.session_state.events))
                    col2.metric("Touch Line", touch_events)
                    col3.metric("Lane Change", change_events)
                    col4.metric("Turn Signal", signal_events)
                    
                    st.divider()
                    
                    # Display events table
                    for i, event in enumerate(st.session_state.events):
                        with st.expander(f"Event {i+1}: {event.vehicle_id} - {event.event_type} ({event.start_time})"):
                            col1, col2 = st.columns(2)
                            col1.write(f"**Vehicle:** {event.vehicle_id}")
                            col2.write(f"**Type:** {event.event_type}")
                            col1.write(f"**Start:** {event.start_time}")
                            col2.write(f"**End:** {event.end_time}")
                            if event.signal_direction:
                                col1.write(f"**Signal:** {event.signal_direction}")
                    
                    # JSON viewer
                    st.subheader("Raw JSON")
                    try:
                        with open(st.session_state.events_json) as f:
                            events_json = json.load(f)
                        st.json(events_json)
                    except Exception as e:
                        st.error(f"Error loading JSON: {e}")
                else:
                    st.info("No events detected in this video.")
            
            with result_tabs[2]:  # Lane Detection tab
                st.subheader("Lane Detection Visualization")
                try:
                    if st.session_state.lane_vis:
                        img = cv2.imread(st.session_state.lane_vis)
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        st.image(img_rgb, use_column_width=True)
                    else:
                        st.info("No lane visualization available.")
                except Exception as e:
                    st.error(f"Error displaying image: {e}")
    
    with tabs[3]:  # About tab
        st.header("About This System")
        st.markdown("""
        ### Architecture Overview
        
        The system uses a multi-stage pipeline:
        
        1. **Lane Detection** - Uses Canny edge detection + Hough transform
        2. **Vehicle Detection** - YOLOv5 for object detection
        3. **Event Detection** - Rule-based analysis for violations
        4. **Visualization** - OpenCV for output generation
        
        ### Supported Event Types
        
        - **touch_line**: Vehicle touches or crosses a lane boundary
        - **change_lane**: Vehicle moves from one lane to another
        - **turn_signal**: Turn signal (left/right) detected
        
        ### Limitations
        
        - Works best on highway/clear road scenarios
        - Requires visible lane markings
        - Turn signal detection is heuristic-based
        - Performance depends on video quality and lighting
        
        ### Technical Stack
        
        - **Python 3.8+**
        - **OpenCV** - Computer vision
        - **YOLOv5** - Vehicle detection
        - **Streamlit** - Web interface
        
        ### Contact & Support
        
        For issues or improvements, please refer to the documentation.
        """)


if __name__ == "__main__":
    create_ui()
