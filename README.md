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
pip install opencv-contrib-python pillow
```

## Running the Application

Simply execute the Python script:

```bash
python app.py
```

---

## **Conclusion**

- For this task, the best tracking algorithm to use is CSRT. 
---

 **Why?** : CSRT is the best balance of accuracy and performance for real-time tracking with a webcam.

- **✔ High Accuracy** – Better than KCF & MOSSE.
- **✔ Handles Occlusions** – Recovers lost objects.
- **✔ No Deep Learning Needed** – Works without AI models.
- **✔ Runs on CPU** – No GPU required.
- **✔ Easy to Use** – Built into OpenCV.
---

**Why Not Others?**

- **KCF** – Faster but less accurate.
- **MOSSE** – Fastest but weak with occlusions.
- **MIL** – Struggles with object disappearance.
- **ByteTrack/DeepSORT** – Overkill for a single object.
- **MedianFlow** – Fails with fast-moving objects.
