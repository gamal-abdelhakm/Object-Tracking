import tkinter as tk
import cv2
from PIL import Image, ImageTk

# Initialize main Tkinter window
tk_window = tk.Tk()
tk_window.title("Object Tracking")
tk_window.configure(bg="#101820")  # Background color
tk_window.geometry("750x600")

# Dictionary of available tracker methods
tracker_types = {
    "MIL": cv2.TrackerMIL_create,
    "KCF": cv2.TrackerKCF_create,
    "CSRT": cv2.TrackerCSRT_create
}

# Variable to hold the selected tracker type; default is "CSRT"
selected_tracker_type = tk.StringVar(value="CSRT")

# Initialize video capture and tracking variables
cap = cv2.VideoCapture(0)
tracker = None
bbox = None
tracking = False

# Video Label
video_label = tk.Label(tk_window, bg="black")
video_label.pack(fill="both", expand=True, padx=10, pady=10)

# Status Label
status_label = tk.Label(tk_window, text="Status: Idle", font=("Arial", 15, "bold"), fg="white", bg="#101820")
status_label.pack(pady=5)

def select_roi():
    global bbox, tracker, tracking
    ret, frame = cap.read()
    if ret:
        cv2.putText(frame, "Select ROI then press 'ESC'", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")
        if bbox != (0, 0, 0, 0):
            tracker_function = tracker_types[selected_tracker_type.get()]
            tracker = tracker_function()
            tracker.init(frame, bbox)
            tracking = True
            status_label.config(text="Status: Tracking", fg="lime")
            

def start_tracking():
    global tracking
    if tracker:
        tracking = True
        status_label.config(text="Status: Tracking", fg="lime")

def stop_tracking():
    global tracking
    tracking = False
    status_label.config(text="Status: Stopped", fg="red")


def update():
    ret, frame = cap.read()
    if ret:
        if tracking and tracker:
            success, new_bbox = tracker.update(frame)
            if success:
                x, y, w, h = map(int, new_bbox)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Tracking", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                status_label.config(text="Status: Lost Tracking", fg="orange")

        # Convert frame to Tkinter format
        img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        video_label.imgtk = img
        video_label.configure(image=img)
    tk_window.after(15, update)

# Unified Grid Layout
control_frame = tk.Frame(tk_window, bg="#2C3E50")
control_frame.pack(padx=10, pady=10)

tk.Label(control_frame, text="Select Tracker:", font=("Arial", 12, "bold"), fg="white", bg="#2C3E50").grid(row=0, column=0, padx=5, pady=5)
tracker_menu = tk.ttk.Combobox(control_frame, textvariable=selected_tracker_type, values=list(tracker_types.keys()), state="readonly")
tracker_menu.grid(row=0, column=1, padx=5, pady=5)

tk.Button(control_frame, text="Select ROI", width=15, command=select_roi, bg="#3498DB", fg="white", font=("Arial", 10, "bold")).grid(row=0, column=2, padx=5, pady=5)
tk.Button(control_frame, text="Start Tracking", width=15, command=start_tracking, bg="#2ECC71", fg="white", font=("Arial", 10, "bold")).grid(row=0, column=3, padx=5, pady=5)
tk.Button(control_frame, text="Stop Tracking", width=15, command=stop_tracking, bg="#E74C3C", fg="white", font=("Arial", 10, "bold")).grid(row=0, column=4, padx=5, pady=5)

# Start updating the video feed
update()
tk_window.mainloop()
cap.release()
cv2.destroyAllWindows()