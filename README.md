# Object Tracking App

## Overview

This application provides real-time object tracking using OpenCV and Tkinter for a simple GUI interface. The user can select an object to track using various tracking algorithms, start or stop tracking, and monitor the live feed with visual feedback.

## Implementation Details

- **GUI Framework:** Tkinter is used to create the user interface.
- **Video Processing:** OpenCV captures frames from the webcam.
- **Tracking Algorithms:** The user can choose from multiple tracking methods (MIL, KCF, CSRT).
- **Real-time Updates:** The application continuously updates the video feed with tracking results.
- **User Controls:** Options to select a region of interest (ROI), start tracking, and stop tracking.

## How It Works

1. Run the script to launch the application.
2. Choose a tracking algorithm from the dropdown menu.
3. Click "Select Area" to define the object to track.
4. Start tracking to see real-time object movement.
5. Stop tracking if needed.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- PIL (`Pillow`)
- Tkinter (comes with Python)

## Installation

Ensure you have the required dependencies installed:

```bash
pip install opencv-python pillow
```

## Running the Application

Simply execute the Python script:

```bash
python app.py
```

