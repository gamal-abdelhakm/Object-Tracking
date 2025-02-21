import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk

# Setup window
win = tk.Tk()
win.title("Object Tracking")
win.configure(bg="#101820")
win.geometry("750x600")

# Tracker options
types = {"MIL": cv2.TrackerMIL_create, "KCF": cv2.TrackerKCF_create, "CSRT": cv2.TrackerCSRT_create}
tracker_type = tk.StringVar(value="CSRT")

# Video capture
cap = cv2.VideoCapture(0)
tracker, bbox, tracking = None, None, False

# UI Elements
video_lbl = tk.Label(win, bg="black")
video_lbl.pack(fill="both", expand=True, padx=10, pady=10)
status_lbl = tk.Label(win, text="Idle", font=("Arial", 15, "bold"), fg="white", bg="#101820")
status_lbl.pack(pady=5)

def select_roi():
    global bbox, tracker, tracking
    ret, frame = cap.read()
    if ret:
        cv2.putText(frame, "Select ROI & press ESC", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")
        if bbox != (0, 0, 0, 0):
            tracker = types[tracker_type.get()]()
            tracker.init(frame, bbox)
            tracking = True
            status_lbl.config(text="Status: Tracking", fg="lime")

def start():
    global tracking
    tracking = True
    status_lbl.config(text="Status: Tracking", fg="lime")

def stop():
    global tracking
    tracking = False
    status_lbl.config(text="Status: Stopped", fg="red")

def update():
    ret, frame = cap.read()
    if ret:
        if tracking and tracker:
            success, new_bbox = tracker.update(frame)
            if success:
                x, y, w, h = map(int, new_bbox)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                status_lbl.config(text="Status: Lost Tracking", fg="orange")
        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        video_lbl.imgtk = img
        video_lbl.configure(image=img)
    win.after(15, update)

# Controls
frame = tk.Frame(win, bg="#2C3E50")
frame.pack(padx=10, pady=10)
tk.Label(frame, text="Select Tracker:", font=("Arial", 12, "bold"), fg="white", bg="#2C3E50").grid(row=0, column=0, padx=5, pady=5)
tt = ttk.Combobox(frame, textvariable=tracker_type, values=list(types.keys()), state="readonly")
tt.grid(row=0, column=1, padx=5, pady=5)
tk.Button(frame, text="Select ROI", width=12, command=select_roi, bg="#3498DB", fg="white", font=("Arial", 10, "bold")).grid(row=0, column=2, padx=5, pady=5)
tk.Button(frame, text="Start Tracking", width=12, command=start, bg="#2ECC71", fg="white", font=("Arial", 10, "bold")).grid(row=0, column=3, padx=5, pady=5)
tk.Button(frame, text="Stop Tracking", width=12, command=stop, bg="#E74C3C", fg="white", font=("Arial", 10, "bold")).grid(row=0, column=4, padx=5, pady=5)

# Main loop
update()
win.mainloop()
cap.release()
cv2.destroyAllWindows()
