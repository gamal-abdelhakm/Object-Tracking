import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
from collections import deque
import numpy as np
import time

class ObjectTrackingApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Object Tracking")
        self.window.configure(bg="#101820")
        self.window.geometry("950x750")

        # Initialize variables
        self.types = {
            "CSRT": cv2.legacy.TrackerCSRT_create,
            "KCF": cv2.legacy.TrackerKCF_create,
            "MOSSE": cv2.legacy.TrackerMOSSE_create,
            "MedianFlow": cv2.legacy.TrackerMedianFlow_create,
            "BOOSTING": cv2.legacy.TrackerBoosting_create
        }
        self.tracker_type = tk.StringVar(value="CSRT")
        self.show_trajectory = tk.BooleanVar(value=True)
        self.max_points = tk.IntVar(value=60)
        self.trajectory = deque(maxlen=self.max_points.get())
        
        # Tracking quality variables
        self.confidence_threshold = tk.DoubleVar(value=0.5)
        self.consecutive_failures = 0
        self.max_consecutive_failures = tk.IntVar(value=10)
        self.last_known_bbox = None
        self.last_positions = deque(maxlen=5)  # Store recent positions for smoothing
        
        self.initial_roi_features = None
        self.orb = cv2.ORB_create()
        
        # Initialize camera
        self.cap = None
        self.camera_index = tk.IntVar(value=0)
        self.open_camera()
        
        self.tracker = None
        self.bbox = None
        self.tracking = False
        self.kalman_filter = None
        
        self.use_smoothing = tk.BooleanVar(value=True)
        self.use_kalman = tk.BooleanVar(value=True)
        
        self.setup_ui()
        self.update()

    def open_camera(self):
        if self.cap is not None:
            self.cap.release()
        
        try:
            self.cap = cv2.VideoCapture(self.camera_index.get())
            if not self.cap.isOpened():
                messagebox.showerror("Error", f"Could not open camera {self.camera_index.get()}")
                self.cap = cv2.VideoCapture(0)  # Try default camera
        except Exception as e:
            messagebox.showerror("Camera Error", f"Error opening camera: {e}")
            self.cap = cv2.VideoCapture(0)  # Try default camera

    def setup_ui(self):
        # Main layout with two frames
        main_frame = tk.Frame(self.window, bg="#101820")
        main_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Left frame for video
        video_frame = tk.Frame(main_frame, bg="#101820")
        video_frame.pack(side=tk.LEFT, fill="both", expand=True)
        
        # Video display
        self.video_lbl = tk.Label(video_frame, bg="black")
        self.video_lbl.pack(fill="both", expand=True, padx=5, pady=5)

        # Status label
        self.status_lbl = tk.Label(video_frame, text="Idle", font=("Arial", 12, "bold"), 
                                  fg="white", bg="#101820")
        self.status_lbl.pack(pady=(0, 5))

        # Right frame for controls
        control_frame = tk.Frame(main_frame, bg="#2C3E50", padx=10, pady=10, width=250)
        control_frame.pack(side=tk.RIGHT, fill="y", padx=(10, 5), pady=5)
        control_frame.pack_propagate(False)
        
        # Tracker settings
        tk.Label(control_frame, text="TRACKER SETTINGS", font=("Arial", 12, "bold"), 
                fg="#3498DB", bg="#2C3E50").pack(anchor="w", pady=(0, 10))
        
        # Tracker selection
        tracker_frame = tk.Frame(control_frame, bg="#2C3E50")
        tracker_frame.pack(fill="x", pady=5)
        tk.Label(tracker_frame, text="Tracker Type:", font=("Arial", 10), 
                fg="white", bg="#2C3E50").pack(side=tk.LEFT)
        tt = ttk.Combobox(tracker_frame, textvariable=self.tracker_type, 
                         values=list(self.types.keys()), state="readonly", width=15)
        tt.pack(side=tk.RIGHT)
        tt.bind("<<ComboboxSelected>>", self.on_tracker_change)
        
        # Confidence threshold
        conf_frame = tk.Frame(control_frame, bg="#2C3E50")
        conf_frame.pack(fill="x", pady=5)
        tk.Label(conf_frame, text="Confidence Threshold:", font=("Arial", 10),
                fg="white", bg="#2C3E50").pack(anchor="w")
        tk.Scale(conf_frame, from_=0.1, to=0.9, resolution=0.1, variable=self.confidence_threshold,
                orient=tk.HORIZONTAL, bg="#2C3E50", fg="white").pack(fill="x")
                
        # Trajectory settings
        tk.Label(control_frame, text="TRAJECTORY SETTINGS", font=("Arial", 12, "bold"),
                fg="#3498DB", bg="#2C3E50").pack(anchor="w", pady=(10, 5))
        
        # Show trajectory checkbox
        tk.Checkbutton(control_frame, text="Show Trajectory", variable=self.show_trajectory,
                      bg="#2C3E50", fg="white", selectcolor="#2C3E50",
                      activebackground="#2C3E50", activeforeground="white").pack(anchor="w")
        
        # Trail length
        trail_frame = tk.Frame(control_frame, bg="#2C3E50")
        trail_frame.pack(fill="x", pady=5)
        tk.Label(trail_frame, text="Trail Length:", fg="white", bg="#2C3E50").pack(anchor="w")
        tk.Scale(trail_frame, from_=10, to=200, variable=self.max_points, 
                orient=tk.HORIZONTAL, bg="#2C3E50", fg="white",
                command=self.update_trail_length).pack(fill="x")
        
        # Advanced Settings
        tk.Label(control_frame, text="ADVANCED SETTINGS", font=("Arial", 12, "bold"),
                fg="#3498DB", bg="#2C3E50").pack(anchor="w", pady=(10, 5))
        
        # Smoothing checkbox
        tk.Checkbutton(control_frame, text="Apply Smoothing", variable=self.use_smoothing,
                      bg="#2C3E50", fg="white", selectcolor="#2C3E50",
                      activebackground="#2C3E50", activeforeground="white").pack(anchor="w")
        
        # Kalman filtering checkbox
        tk.Checkbutton(control_frame, text="Apply Kalman Filter", variable=self.use_kalman,
                      bg="#2C3E50", fg="white", selectcolor="#2C3E50",
                      activebackground="#2C3E50", activeforeground="white").pack(anchor="w")
        
        # Max consecutive failures
        fail_frame = tk.Frame(control_frame, bg="#2C3E50")
        fail_frame.pack(fill="x", pady=5)
        tk.Label(fail_frame, text="Max Recovery Attempts:", fg="white", bg="#2C3E50").pack(anchor="w")
        tk.Scale(fail_frame, from_=5, to=30, variable=self.max_consecutive_failures,
                orient=tk.HORIZONTAL, bg="#2C3E50", fg="white").pack(fill="x")
        
        # Action Buttons
        tk.Label(control_frame, text="ACTIONS", font=("Arial", 12, "bold"),
                fg="#3498DB", bg="#2C3E50").pack(anchor="w", pady=(10, 5))
        
        # Buttons
        tk.Button(control_frame, text="Select ROI", width=12, command=self.select_roi,
                 bg="#3498DB", fg="white", font=("Arial", 10, "bold")).pack(fill="x", pady=5)
        tk.Button(control_frame, text="Start Tracking", width=12, command=self.start,
                 bg="#2ECC71", fg="white", font=("Arial", 10, "bold")).pack(fill="x", pady=5)
        tk.Button(control_frame, text="Stop Tracking", width=12, command=self.stop,
                 bg="#E74C3C", fg="white", font=("Arial", 10, "bold")).pack(fill="x", pady=5)
        tk.Button(control_frame, text="Reset", width=12, command=self.reset,
                 bg="#F39C12", fg="white", font=("Arial", 10, "bold")).pack(fill="x", pady=5)

    def update_trail_length(self, val):
        # Update trajectory deque size when slider is moved
        new_length = self.max_points.get()
        temp = list(self.trajectory)
        self.trajectory = deque(temp[-new_length:] if len(temp) > new_length else temp, maxlen=new_length)

    def on_tracker_change(self, event=None):
        # Reset tracker if type is changed
        if self.tracking:
            self.init_tracker(self.last_frame)

    def init_kalman_filter(self, bbox):
        # Initialize Kalman filter for smoother tracking
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
        
        # Initialize with current bbox
        x, y, w, h = bbox
        center_x = x + w/2
        center_y = y + h/2
        
        kalman.statePre = np.array([[center_x], [center_y], [0], [0]], np.float32)
        kalman.statePost = np.array([[center_x], [center_y], [0], [0]], np.float32)
        
        return kalman

    def select_roi(self):
        ret, frame = self.cap.read()
        if ret:
            self.last_frame = frame.copy()
            cv2.putText(frame, "Select ROI & press ENTER", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            self.bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select ROI")
            if self.bbox != (0, 0, 0, 0):
                # Store initial ROI features
                x, y, w, h = self.bbox
                roi = frame[y:y+h, x:x+w]
                _, self.initial_roi_features = self.orb.detectAndCompute(roi, None)
                self.init_tracker(frame)
                self.trajectory.clear()
                self.last_positions.clear()
                self.status_lbl.config(text="Status: Tracking", fg="lime")

    def compare_features(self, current_roi):
        # Compare features between initial ROI and current ROI
        try:
            # Detect features in current ROI
            _, current_features = self.orb.detectAndCompute(current_roi, None)
            
            if current_features is None or self.initial_roi_features is None:
                return False
            
            # Create BF matcher and match features
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(self.initial_roi_features, current_features)
            
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Calculate similarity score based on number and quality of matches
            if len(matches) > 10:  # Minimum number of matches threshold
                avg_distance = sum(m.distance for m in matches[:10]) / 10
                return avg_distance < 50  # Distance threshold
            return False
        except Exception:
            return False

    def init_tracker(self, frame):
        # Initialize tracker with current frame and bbox
        self.tracker = self.types[self.tracker_type.get()]()
        self.tracker.init(frame, self.bbox)
        self.tracking = True
        self.last_known_bbox = self.bbox
        self.consecutive_failures = 0
        
        # Initialize Kalman filter
        self.kalman_filter = self.init_kalman_filter(self.bbox)

    def start(self):
        if self.bbox is not None:
            ret, frame = self.cap.read()
            if ret:
                self.init_tracker(frame)
                self.status_lbl.config(text="Status: Tracking", fg="lime")

    def stop(self):
        self.tracking = False
        self.status_lbl.config(text="Status: Stopped", fg="red")

    def reset(self):
        self.tracking = False
        self.bbox = None
        self.tracker = None
        self.last_known_bbox = None
        self.trajectory.clear()
        self.last_positions.clear()
        self.consecutive_failures = 0
        self.status_lbl.config(text="Status: Reset", fg="white")

    def predict_position(self):
        # Use Kalman filter to predict next position
        prediction = self.kalman_filter.predict()
        center_x, center_y = prediction[0, 0], prediction[1, 0]
        
        # Convert center point to bbox
        w, h = self.last_known_bbox[2], self.last_known_bbox[3]
        x = center_x - w/2
        y = center_y - h/2
        
        return (int(x), int(y), int(w), int(h))

    def update_kalman(self, bbox):
        # Update Kalman filter with new measurement
        x, y, w, h = bbox
        center_x = x + w/2
        center_y = y + h/2
        
        measurement = np.array([[center_x], [center_y]], np.float32)
        self.kalman_filter.correct(measurement)

    def apply_smoothing(self, bbox):
        # Apply smoothing to reduce jitter
        x, y, w, h = bbox
        
        # Add current position to history
        self.last_positions.append((x, y))
        
        # If we have enough positions, use moving average
        if len(self.last_positions) >= 3:
            smooth_x = sum(pos[0] for pos in self.last_positions) / len(self.last_positions)
            smooth_y = sum(pos[1] for pos in self.last_positions) / len(self.last_positions)
            return (int(smooth_x), int(smooth_y), w, h)
        
        return bbox

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            self.last_frame = frame.copy()       
            if self.tracking and self.tracker:
                # Try to update tracker
                success, new_bbox = self.tracker.update(frame)
                
                # Check if tracking is still reliable
                if success:
                    # Apply Kalman filtering for prediction
                    if self.use_kalman.get() and self.kalman_filter is not None:
                        # Update Kalman with new measurement
                        self.update_kalman(new_bbox)
                        # Get prediction
                        predicted_bbox = self.predict_position()
                        
                        # Use prediction or actual depending on confidence
                        if self.consecutive_failures > 0:
                            # Blend predicted and actual positions
                            blend_factor = min(self.consecutive_failures / self.max_consecutive_failures.get(), 0.8)
                            x1, y1, w1, h1 = predicted_bbox
                            x2, y2, w2, h2 = new_bbox
                            blended_bbox = (
                                int(x1 * blend_factor + x2 * (1 - blend_factor)),
                                int(y1 * blend_factor + y2 * (1 - blend_factor)),
                                w2, h2
                            )
                            new_bbox = blended_bbox
                        
                    # Apply smoothing to reduce jitter if enabled
                    if self.use_smoothing.get():
                        new_bbox = self.apply_smoothing(new_bbox)
                    
                    x, y, w, h = map(int, new_bbox)
                    self.last_known_bbox = (x, y, w, h)
                    self.consecutive_failures = 0
                    
                    # Draw rectangle and text
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Tracking", (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Calculate and store center point for trajectory
                    if self.show_trajectory.get():
                        center = (int(x + w/2), int(y + h/2))
                        self.trajectory.append(center)
                        
                        # Draw trajectory
                        if len(self.trajectory) > 1:
                            for i in range(1, len(self.trajectory)):
                                thickness = int(np.sqrt(64 / float(i + 1)) * 1.5)
                                cv2.line(frame, self.trajectory[i-1], self.trajectory[i], 
                                        (0, 0, 255), thickness)
                else:
                    # Tracking failed
                    self.consecutive_failures += 1
                    
                    if self.consecutive_failures < self.max_consecutive_failures.get():
                        # Try to recover using Kalman prediction
                        if self.kalman_filter is not None and self.last_known_bbox is not None:
                            predicted_bbox = self.predict_position()
                            x, y, w, h = map(int, predicted_bbox)
                            
                            # Draw rectangle with different color to indicate prediction
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                            cv2.putText(frame, "Predicted", (x, y - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                            
                            self.status_lbl.config(text=f"Status: Recovering ({self.consecutive_failures})", fg="orange")
                    else:
                        # Too many consecutive failures, reset tracker
                        if self.last_known_bbox is not None:
                            # Extract current ROI using last known bbox
                            x, y, w, h = self.last_known_bbox
                            current_roi = frame[max(0, y):min(frame.shape[0], y+h), 
                                                max(0, x):min(frame.shape[1], x+w)]
                            
                            # Check feature similarity
                            if self.compare_features(current_roi):
                                # Features match, reinitialize tracker
                                self.bbox = self.last_known_bbox
                                self.init_tracker(frame)
                                self.status_lbl.config(text="Status: Reinitialized", fg="yellow")
                            else:
                                # Features don't match, signal track loss
                                self.status_lbl.config(text="Status: Object Changed/Lost", fg="red")
                                self.reset()
            
            # Convert to RGB for display
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(image=img)
            
            self.video_lbl.imgtk = img_tk
            self.video_lbl.configure(image=img_tk)
        
        self.window.after(33, self.update)

    def cleanup(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

# run method to run the application
def run():
    root = tk.Tk()
    app = ObjectTrackingApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.cleanup(), root.destroy()))
    root.mainloop()

if __name__ == "__main__":
    run()