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
