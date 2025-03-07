import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
from collections import deque
import streamlit as st

class ObjectTrackingApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Object Tracking")
        self.window.configure(bg="#101820")
        self.window.geometry("850x700")

        # Initialize variables
        self.types = {
            "CSRT": cv2.legacy.TrackerCSRT_create,
            "KCF": cv2.legacy.TrackerKCF_create,
            "MOSSE": cv2.legacy.TrackerMOSSE_create
        }
        self.tracker_type = tk.StringVar(value="CSRT")
        self.show_trajectory = tk.BooleanVar(value=False)
        self.max_points = tk.IntVar(value=30)
        self.trajectory = deque(maxlen=self.max_points.get())
        
        self.cap = cv2.VideoCapture(0)
        self.tracker = None
        self.bbox = None
        self.tracking = False

        self.setup_ui()
        self.update()

    def setup_ui(self):
        # Video display
        self.video_lbl = tk.Label(self.window, bg="black")
        self.video_lbl.pack(fill="both", expand=True, padx=10)

        # Status label
        self.status_lbl = tk.Label(self.window, text="Idle", font=("Arial", 15, "bold"), 
                                 fg="white", bg="#101820")
        self.status_lbl.pack()

        # Controls frame
        control_frame = tk.Frame(self.window, bg="#2C3E50")
        control_frame.pack(padx=10)

        # Tracker selection
        tk.Label(control_frame, text="Select Tracker:", font=("Arial", 12, "bold"), 
                fg="white", bg="#2C3E50").grid(row=0, column=0, padx=5, pady=5)
        tt = ttk.Combobox(control_frame, textvariable=self.tracker_type, 
                         values=list(self.types.keys()), state="readonly")
        tt.grid(row=0, column=1, padx=5)

        # Trajectory controls
        tk.Checkbutton(control_frame, text="Show Trajectory", variable=self.show_trajectory,
                      bg="#2C3E50", fg="white").grid(row=0, column=2, padx=5)
        tk.Label(control_frame, text="Trail Length:", fg="white", bg="#2C3E50").grid(row=0, column=3)
        tk.Scale(control_frame, from_=10, to=100, variable=self.max_points, 
                orient=tk.HORIZONTAL, bg="#2C3E50", fg="white").grid(row=0, column=4)

        # Buttons
        buttons_frame = tk.Frame(self.window, bg="#101820")
        buttons_frame.pack(pady=5)
        
        tk.Button(buttons_frame, text="Select ROI", width=12, command=self.select_roi,
                 bg="#3498DB", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        tk.Button(buttons_frame, text="Start Tracking", width=12, command=self.start,
                 bg="#2ECC71", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        tk.Button(buttons_frame, text="Stop Tracking", width=12, command=self.stop,
                 bg="#E74C3C", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)

    def select_roi(self):
        ret, frame = self.cap.read()
        if ret:
            cv2.putText(frame, "Select ROI & press ENTER", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            self.bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select ROI")
            if self.bbox != (0, 0, 0, 0):
                self.tracker = self.types[self.tracker_type.get()]()
                self.tracker.init(frame, self.bbox)
                self.tracking = True
                self.trajectory.clear()
                self.status_lbl.config(text="Status: Tracking", fg="lime")

    def start(self):
        if self.bbox is not None:
            self.tracking = True
            self.status_lbl.config(text="Status: Tracking", fg="lime")

    def stop(self):
        self.tracking = False
        self.status_lbl.config(text="Status: Stopped", fg="red")

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            if self.tracking and self.tracker:
                success, new_bbox = self.tracker.update(frame)
                if success:
                    x, y, w, h = map(int, new_bbox)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Tracking", (x, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    if self.show_trajectory.get():
                        center = (int(x + w/2), int(y + h/2))
                        self.trajectory.append(center)
                        for i in range(1, len(self.trajectory)):
                            cv2.line(frame, self.trajectory[i-1], self.trajectory[i], 
                                   (0, 0, 255), 2)
                else:
                    self.status_lbl.config(text="Status: Lost Tracking", fg="orange")

            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
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
