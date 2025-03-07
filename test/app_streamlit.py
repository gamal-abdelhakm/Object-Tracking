import streamlit as st
import cv2
import numpy as np
from collections import deque

# Basic page config
st.set_page_config(page_title="Simple Object Tracking", layout="wide")

# Available trackers
TRACKERS = {
    "CSRT": cv2.legacy.TrackerCSRT_create,
    "KCF": cv2.legacy.TrackerKCF_create,
    "MOSSE": cv2.legacy.TrackerMOSSE_create,
}

# Sidebar controls
st.sidebar.title("Settings")
tracker_type = st.sidebar.selectbox("Select Tracker", list(TRACKERS.keys()))
show_trajectory = st.sidebar.checkbox("Show Trajectory", True)
max_points = st.sidebar.slider("Trajectory Length", 10, 100, 30)

# Main page
st.title("Simple Object Tracking")

if st.button("Start Tracking"):
    st.info("Tracking window will open in a separate window. Press 'Q' to quit.")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        st.error("Error: Could not access webcam")
        st.stop()

    # Select ROI
    bbox = cv2.selectROI("Select Object", frame)
    cv2.destroyAllWindows()
    
    # Initialize tracker
    tracker = TRACKERS[tracker_type]()
    tracker.init(frame, bbox)

    # Initialize trajectory points
    trajectory = deque(maxlen=max_points)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update tracker
        success, bbox = tracker.update(frame)

        if success:
            # Draw tracking box
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Update trajectory
            if show_trajectory:
                center = (int(x + w/2), int(y + h/2))
                trajectory.append(center)
                
                # Draw trajectory
                for i in range(1, len(trajectory)):
                    cv2.line(frame, trajectory[i-1], trajectory[i], (0, 0, 255), 2)

            # Add status text
            cv2.putText(frame, "Tracking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Lost", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 0, 255), 2)

        # Show frame in OpenCV window
        cv2.imshow("Tracking", frame)

        # Exit if Q pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

# Help section
with st.expander("Help"):
    st.markdown("""
    # Advanced Object Tracking Application

    A real-time object tracking application built with Python, OpenCV, and Tkinter. The application provides multiple tracking algorithms and advanced features like Kalman filtering and trajectory visualization.

    ## Features

    - Multiple tracking algorithms (CSRT, KCF, MOSSE)
    - Kalman filter for prediction and smoothing
    - Real-time trajectory visualization
    - Auto-recovery when tracking is lost
    - Performance metrics (FPS, confidence score, tracking time)
    - User-friendly GUI with dark theme
    - Configurable settings

    ## Requirements

    ```python
    opencv-python
    numpy
    pillow
    tkinter
    ```

    ## Usage

    1. Run the application:
    ```bash
    python app_tkinter.py
    ```

    2. Select a tracking algorithm from the dropdown menu
    3. Click "Select ROI" and draw a box around the object you want to track
    4. Press ENTER to confirm selection
    5. Use the control buttons to start/stop tracking

    ## Advanced Settings

    - **Show Trajectory**: Toggle trajectory visualization
    - **Use Kalman Filter**: Enable/disable Kalman filter predictions
    - **Auto Recovery**: Automatically attempt to recover lost tracking
    - **Trail Length**: Adjust the length of the trajectory trail
    - **Recovery Attempts**: Set maximum number of recovery attempts

    ## Controls

    - **Select ROI**: Choose the object to track
    - **Start Tracking**: Begin tracking the selected object
    - **Stop Tracking**: Pause the tracking
    - **Reset**: Clear current tracking and settings

    ## Performance

    The application displays real-time metrics:
    - FPS (Frames Per Second)
    - Tracking confidence score
    - Total tracking time
    """)