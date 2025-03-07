import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from collections import deque

class KalmanFilter:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 1
        self.initialized = False

    def init(self, x, y):
        self.kalman.statePre = np.array([[x], [y], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[x], [y], [0], [0]], np.float32)
        self.initialized = True

    def predict(self):
        if not self.initialized:
            return None
        prediction = self.kalman.predict()
        return int(prediction[0]), int(prediction[1])

    def correct(self, x, y):
        measurement = np.array([[x], [y]], np.float32)
        self.kalman.correct(measurement)

class ObjectTrackingApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Advanced Object Tracking")
        self.window.configure(bg="#1E1E2E")
        self.window.geometry("900x750")

        # Create a custom style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TButton', font=('Arial', 10, 'bold'), background='#6C7B8B')
        self.style.configure('TCombobox', foreground='#333333', background='#6C7B8B')
        self.style.configure('TCheckbutton', foreground='white', background='#2C3E50')
        self.style.configure('TScale', background='#2C3E50', troughcolor='#6C7B8B')

        # Initialize variables
        self.types = {
            "CSRT": cv2.legacy.TrackerCSRT_create,
            "KCF": cv2.legacy.TrackerKCF_create,
            "MOSSE": cv2.legacy.TrackerMOSSE_create
        }
        self.tracker_type = tk.StringVar(value="CSRT")
        self.show_trajectory = tk.BooleanVar(value=True)
        self.use_kalman = tk.BooleanVar(value=True)
        self.auto_recover = tk.BooleanVar(value=False)
        self.max_points = tk.IntVar(value=40)
        self.trajectory = deque(maxlen=self.max_points.get())
        
        self.cap = cv2.VideoCapture(0)
        self.tracker = None
        self.kalman = KalmanFilter()
        self.bbox = None
        self.tracking = False
        self.lost_count = 0
        self.last_bbox = None
        self.recovery_attempts = 0
        self.max_recovery_attempts = 5

        # Add tracking metrics
        self.fps_avg = 0
        self.confidence_score = 0
        self.tracking_time = 0
        self.start_time = None

        self.setup_ui()
        self.update()

    def setup_ui(self):
        # Main container for better organization
        main_container = tk.Frame(self.window, bg="#1E1E2E")
        main_container.pack(fill="both", expand=True, padx=10)

        # Top frame for title and stats
        top_frame = tk.Frame(main_container, bg="#1E1E2E")
        top_frame.pack(fill="x")
        
        tk.Label(top_frame, text="Advanced Object Tracking", font=("Arial", 16, "bold"), 
                fg="#F8F8F2", bg="#1E1E2E").pack(side=tk.LEFT, padx=10)
        
        self.stats_frame = tk.Frame(top_frame, bg="#1E1E2E")
        self.stats_frame.pack(side=tk.RIGHT, padx=10)
        
        self.fps_label = tk.Label(self.stats_frame, text="FPS: 0", font=("Arial", 10), 
                               fg="#F8F8F2", bg="#1E1E2E")
        self.fps_label.pack(side=tk.LEFT, padx=5)
        
        self.confidence_label = tk.Label(self.stats_frame, text="Confidence: 0%", font=("Arial", 10), 
                                      fg="#F8F8F2", bg="#1E1E2E")
        self.confidence_label.pack(side=tk.LEFT, padx=5)
        
        self.time_label = tk.Label(self.stats_frame, text="Time: 0s", font=("Arial", 10), 
                                fg="#F8F8F2", bg="#1E1E2E")
        self.time_label.pack(side=tk.LEFT, padx=5)

        # Create a frame for the video with a border
        video_frame = tk.Frame(main_container, bg="#44475A", bd=2, relief=tk.RIDGE)
        video_frame.pack(fill="both", expand=True, padx=10)
        
        # Video display
        self.video_lbl = tk.Label(video_frame, bg="black")
        self.video_lbl.pack(fill="both", expand=True, padx=2)

        # Status label
        status_frame = tk.Frame(main_container, bg="#1E1E2E")
        status_frame.pack(fill="x")
        
        self.status_lbl = tk.Label(status_frame, text="Status: Idle", font=("Arial", 12, "bold"), 
                                 fg="#F8F8F2", bg="#1E1E2E")
        self.status_lbl.pack(side=tk.LEFT, padx=10)
        
        # Notification label for recovery attempts
        self.notification_lbl = tk.Label(status_frame, text="", font=("Arial", 10), 
                                      fg="#F8F8F2", bg="#1E1E2E")
        self.notification_lbl.pack(side=tk.RIGHT, padx=10)

        # Controls frame with tabbed interface
        control_frame = ttk.Notebook(main_container)
        control_frame.pack(fill="x", padx=10)
        
        # Basic controls tab
        basic_tab = tk.Frame(control_frame, bg="#282A36")
        control_frame.add(basic_tab, text="Basic Controls")
        
        # Tracker selection
        tracker_frame = tk.Frame(basic_tab, bg="#282A36")
        tracker_frame.pack(fill="x")
        
        tk.Label(tracker_frame, text="Select Tracker:", font=("Arial", 11), 
                fg="#F8F8F2", bg="#282A36").pack(side=tk.LEFT, padx=10)
        
        tt = ttk.Combobox(tracker_frame, textvariable=self.tracker_type, 
                         values=list(self.types.keys()), state="readonly", width=10)
        tt.pack(side=tk.LEFT, padx=5)
        
        # Buttons
        buttons_frame = tk.Frame(basic_tab, bg="#282A36")
        buttons_frame.pack(fill="x")
        
        ttk.Button(buttons_frame, text="Select ROI", width=15, command=self.select_roi, 
                 style="TButton").pack(side=tk.LEFT, padx=10, pady=5)
        
        ttk.Button(buttons_frame, text="Start Tracking", width=15, command=self.start, 
                 style="TButton").pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Button(buttons_frame, text="Stop Tracking", width=15, command=self.stop, 
                 style="TButton").pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Button(buttons_frame, text="Reset", width=15, command=self.reset, 
                 style="TButton").pack(side=tk.LEFT, padx=5, pady=5)

        # Advanced controls tab
        advanced_tab = tk.Frame(control_frame, bg="#282A36")
        control_frame.add(advanced_tab, text="Advanced Settings")
        
        # Left frame for checkboxes
        check_frame = tk.Frame(advanced_tab, bg="#282A36")
        check_frame.pack(side=tk.LEFT, padx=20, fill="y")
        
        ttk.Checkbutton(check_frame, text="Show Trajectory", variable=self.show_trajectory, 
                       style="TCheckbutton").pack(anchor="w")
        
        ttk.Checkbutton(check_frame, text="Use Kalman Filter", variable=self.use_kalman, 
                       style="TCheckbutton").pack(anchor="w")
        
        ttk.Checkbutton(check_frame, text="Auto Recovery", variable=self.auto_recover, 
                       style="TCheckbutton").pack(anchor="w")
        
        # Right frame for sliders
        slider_frame1 = tk.Frame(advanced_tab, bg="#282A36")
        slider_frame1.pack(side=tk.LEFT, padx=20, fill="y")
        
        tk.Label(slider_frame1, text="Trail Length:", fg="#F8F8F2", bg="#282A36").pack(anchor="w")
        tk.Scale(slider_frame1, from_=10, to=100, variable=self.max_points, 
                orient=tk.HORIZONTAL, width=20, length=150, bg="#383A59", fg="#F8F8F2",
                command=self.update_max_points).pack()
        
        slider_frame2 = tk.Frame(advanced_tab, bg="#282A36")
        slider_frame2.pack(side=tk.LEFT, padx=20, fill="y")
        
        tk.Label(slider_frame2, text="Recovery Attempts:", fg="#F8F8F2", bg="#282A36").pack(anchor="w")
        tk.Scale(slider_frame2, from_=1, to=10, variable=self.max_recovery_attempts, 
                orient=tk.HORIZONTAL, width=20, length=150, bg="#383A59", fg="#F8F8F2").pack()

    def update_max_points(self, val):
        # Update the deque maxlen when the slider changes
        new_max = self.max_points.get()
        temp_trajectory = list(self.trajectory)
        self.trajectory = deque(maxlen=new_max)
        for point in temp_trajectory[-new_max:]:
            self.trajectory.append(point)

    def select_roi(self):
        self.stop()  # Stop any current tracking
        ret, frame = self.cap.read()
        if ret:
            cv2.putText(frame, "Select ROI & press ENTER", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press ESC to cancel", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            self.bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select ROI")
            if self.bbox != (0, 0, 0, 0):
                self.tracker = self.types[self.tracker_type.get()]()
                self.tracker.init(frame, self.bbox)
                self.last_bbox = self.bbox
                
                # Initialize Kalman filter with the center of the bbox
                x, y, w, h = map(int, self.bbox)
                self.kalman.init(x + w//2, y + h//2)
                
                self.tracking = True
                self.trajectory.clear()
                self.lost_count = 0
                self.recovery_attempts = 0
                self.start_time = cv2.getTickCount()
                self.status_lbl.config(text="Status: Tracking", fg="#50FA7B")
                self.notification_lbl.config(text="")

    def start(self):
        if self.bbox is not None:
            self.tracking = True
            if self.start_time is None:
                self.start_time = cv2.getTickCount()
            self.status_lbl.config(text="Status: Tracking", fg="#50FA7B")

    def stop(self):
        self.tracking = False
        self.status_lbl.config(text="Status: Stopped", fg="#FF5555")

    def reset(self):
        self.stop()
        self.bbox = None
        self.last_bbox = None
        self.trajectory.clear()
        self.lost_count = 0
        self.recovery_attempts = 0
        self.start_time = None
        self.kalman = KalmanFilter()
        self.status_lbl.config(text="Status: Idle", fg="#F8F8F2")
        self.notification_lbl.config(text="")
        self.fps_label.config(text="FPS: 0")
        self.confidence_label.config(text="Confidence: 0%")
        self.time_label.config(text="Time: 0s")

    def attempt_recovery(self, frame):
        if self.recovery_attempts >= self.max_recovery_attempts:
            self.notification_lbl.config(text=f"Recovery failed after {self.max_recovery_attempts} attempts", fg="#FF5555")
            self.stop()
            return False
        
        self.notification_lbl.config(text=f"Attempting recovery... ({self.recovery_attempts+1}/{self.max_recovery_attempts})", fg="#FFB86C")
        
        # If we have a last known position, try to reinitialize there
        if self.last_bbox is not None:
            # Add some jitter to the last bbox to help find the object again
            x, y, w, h = self.last_bbox
            jitter = 10  # pixels to search around
            search_area = (max(0, int(x-jitter)), max(0, int(y-jitter)), 
                          int(w+jitter*2), int(h+jitter*2))
            
            # Create a new tracker
            self.tracker = self.types[self.tracker_type.get()]()
            self.tracker.init(frame, search_area)
            
            self.recovery_attempts += 1
            return True
        
        return False

    def update(self):
        start_time = cv2.getTickCount()
        ret, frame = self.cap.read()
        
        if ret:
            frame_display = frame.copy()
            
            # Tracking logic
            if self.tracking and self.tracker:
                success, new_bbox = self.tracker.update(frame)
                
                if success:
                    x, y, w, h = map(int, new_bbox)
                    self.last_bbox = new_bbox
                    self.lost_count = 0
                    self.recovery_attempts = 0
                    
                    # Calculate confidence (simplified example)
                    confidence_region = frame[max(0, y):min(y+h, frame.shape[0]), 
                                           max(0, x):min(x+w, frame.shape[1])]
                    if confidence_region.size > 0:
                        # Simple confidence based on color variance
                        self.confidence_score = min(100, int(np.std(confidence_region) * 5))
                    
                    # Update Kalman filter with measured position
                    center_x = x + w//2
                    center_y = y + h//2
                    if self.use_kalman.get():
                        self.kalman.correct(center_x, center_y)
                        predicted = self.kalman.predict()
                        if predicted:
                            # Draw predicted position
                            cv2.circle(frame_display, predicted, 5, (0, 255, 255), -1)
                    
                    # Update trajectory
                    if self.show_trajectory.get():
                        center = (center_x, center_y)
                        self.trajectory.append(center)
                        
                        # Draw trajectory
                        pts = np.array(list(self.trajectory), np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(frame_display, [pts], False, (0, 0, 255), 2)
                    
                    # Draw the bounding box
                    cv2.rectangle(frame_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame_display, f"Tracking: {self.confidence_score}%", (x, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    self.status_lbl.config(text="Status: Tracking", fg="#50FA7B")
                    self.notification_lbl.config(text="")
                    
                else:
                    self.lost_count += 1
                    
                    # Get a prediction from Kalman filter
                    if self.use_kalman.get() and self.kalman.initialized:
                        predicted = self.kalman.predict()
                        if predicted:
                            cv2.circle(frame_display, predicted, 10, (0, 165, 255), 2)
                            cv2.putText(frame_display, "Predicted", predicted, 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    
                    # Try to recover if auto recover is enabled
                    if self.lost_count > 10 and self.auto_recover.get():
                        if self.attempt_recovery(frame):
                            self.status_lbl.config(text="Status: Recovering", fg="#FFB86C")
                        else:
                            self.status_lbl.config(text="Status: Lost Tracking", fg="#FF5555")
                    else:
                        self.status_lbl.config(text="Status: Tracking Lost", fg="#FF5555")
            
            # Calculate FPS
            end_time = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (end_time - start_time)
            self.fps_avg = 0.9 * self.fps_avg + 0.1 * fps  # Smooth the FPS
            
            # Update tracking time
            if self.tracking and self.start_time is not None:
                self.tracking_time = (end_time - self.start_time) / cv2.getTickFrequency()
            
            # Update info labels
            self.fps_label.config(text=f"FPS: {int(self.fps_avg)}")
            self.confidence_label.config(text=f"Confidence: {self.confidence_score}%")
            self.time_label.config(text=f"Time: {self.tracking_time:.1f}s")
            
            # Add info overlay to the frame
            cv2.putText(frame_display, f"FPS: {int(self.fps_avg)}", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display the frame
            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)))
            self.video_lbl.imgtk = img
            self.video_lbl.configure(image=img)
        
        self.window.after(15, self.update)

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectTrackingApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.cleanup(), root.destroy()))
    root.mainloop()