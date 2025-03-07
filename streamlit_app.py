import app
import streamlit as st

# Basic page config
st.set_page_config(page_title="Object Tracking", layout="wide")

# Main page
st.title("Object Tracking")
st.markdown("A real-time object tracking application built with Python, OpenCV, and Tkinter. The application provides multiple tracking algorithms and advanced features like Kalman filtering and trajectory visualization.")
st.markdown("---")

# Control parameters
st.sidebar.header("Tracking Settings")
tracking_algorithm = st.sidebar.selectbox(
    "Tracking Algorithm",
    ["CSRT", "KCF", "MOSSE"],
    index=0
)

# Advanced settings
show_trajectory = st.sidebar.checkbox("Show Trajectory", value=True)
use_kalman = st.sidebar.checkbox("Use Kalman Filter", value=True)
auto_recovery = st.sidebar.checkbox("Auto Recovery", value=True)
trail_length = st.sidebar.slider("Trail Length", 1, 100, 30)
recovery_attempts = st.sidebar.slider("Recovery Attempts", 1, 10, 3)

# Start tracking with parameters
if st.button("Start Tracking"):
    app.main(
        tracker_type=tracking_algorithm,
        show_trajectory=show_trajectory,
        use_kalman=use_kalman,
        auto_recovery=auto_recovery,
        trail_length=trail_length,
        recovery_attempts=recovery_attempts
    )

# Features section
with st.expander("Features"):
    st.markdown("""
    - Multiple tracking algorithms (CSRT, KCF, MOSSE)
    - Kalman filter for prediction and smoothing
    - Real-time trajectory visualization
    - Auto-recovery when tracking is lost
    - Performance metrics (FPS, confidence score, tracking time)
    - User-friendly GUI with dark theme
    - Configurable settings
                """)
    
# Help section
with st.expander("Help"):
    st.markdown("""
    ## Usage
                
    1. Run the application:
    2. Select a tracking algorithm from the dropdown menu
    3. Click "Select ROI" and draw a box around the object you want to track
    4. Press ENTER to confirm selection
    5. Use the control buttons to start/stop tracking
                
    ---

    ## Controls

    - **Select ROI**: Choose the object to track
    - **Start Tracking**: Begin tracking the selected object
    - **Stop Tracking**: Pause the tracking
    - **Reset**: Clear current tracking and settings
                
    ---

    ## Performance

    The application displays real-time metrics:
    - FPS (Frames Per Second)
    - Tracking confidence score
    - Total tracking time
                
    ---
                
    ## Advanced Settings

    - **Show Trajectory**: Toggle trajectory visualization
    - **Use Kalman Filter**: Enable/disable Kalman filter predictions
    - **Auto Recovery**: Automatically attempt to recover lost tracking
    - **Trail Length**: Adjust the length of the trajectory trail
    - **Recovery Attempts**: Set maximum number of recovery attempts
    """)

#Developed and maintained by Gamal Ahemd
with st.expander("Developed and maintained by"):
    st.markdown("""
    ## Gamal Ahmed
    - **Email**: gamal.ahmed.abdelhakm@gmail.com
    - **GitHub**: [gamal-abdelhakm](https://github.com/gamal-abdelhakm) | [Object-Tracking](https://github.com/gamal-abdelhakm/Object-Tracking)
    - **LinkedIn**: [gamal-ahmed](https://www.linkedin.com/in/gamal-ahmed-0a6076235/)
    """)