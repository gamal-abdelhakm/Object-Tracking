import cv2
import numpy as np
import time
from collections import deque
import argparse

class ObjectTrackingConsole:
    def __init__(self, camera_index=0, tracker_type="CSRT", show_trajectory=True, 
                 max_points=20, frame_skip=0, confidence_threshold=0.6):
        # Initialize variables
        self.types = {
            "CSRT": cv2.legacy.TrackerCSRT_create,
            "KCF": cv2.legacy.TrackerKCF_create,
            "MOSSE": cv2.legacy.TrackerMOSSE_create,
            "MedianFlow": cv2.legacy.TrackerMedianFlow_create,
            "BOOSTING": cv2.legacy.TrackerBoosting_create
        }
        self.tracker_type = tracker_type
        self.show_trajectory = show_trajectory
        self.max_points = max_points
        self.trajectory = deque(maxlen=max_points)
        
        # Frame processing variables
        self.frame_skip = frame_skip
        self.frame_count = 0
        
        # Tracking quality variables
        self.confidence_threshold = confidence_threshold
        self.consecutive_failures = 0
        self.max_consecutive_failures = 10
        self.last_known_bbox = None
        self.last_positions = deque(maxlen=5)  # Store recent positions for smoothing
        
        # Performance metrics
        self.fps = 0
        self.process_times = deque(maxlen=30)
        
        # Initialize camera
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Could not open camera {self.camera_index}")
            self.cap = cv2.VideoCapture(0)  # Try default camera
            
        self.tracker = None
        self.bbox = None
        self.tracking = False
        self.kalman_filter = None
        self.last_frame = None

        # Add these new instance variables
        self.initial_features = None
        self.initial_roi_image = None
        self.feature_matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        self.orb = cv2.ORB_create()

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
        print("Select an ROI and press ENTER. ESC to cancel selection.")
        ret, frame = self.cap.read()
        if ret:
            self.last_frame = frame.copy()
            cv2.putText(frame, "Select ROI & press ENTER", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            self.bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select ROI")
            if self.bbox != (0, 0, 0, 0):
                # Store initial features
                _, self.initial_features = self.extract_features(frame, self.bbox)
                self.init_tracker(frame)
                self.trajectory.clear()
                self.last_positions.clear()
                print("Status: Tracking")
                return True
        return False

    def init_tracker(self, frame):
        # Initialize tracker with current frame and bbox
        self.tracker = self.types[self.tracker_type]()
        self.tracker.init(frame, self.bbox)
        self.tracking = True
        self.last_known_bbox = self.bbox
        self.consecutive_failures = 0
        
        # Initialize Kalman filter
        self.kalman_filter = self.init_kalman_filter(self.bbox)

    def stop(self):
        self.tracking = False
        print("Status: Stopped")

    def reset(self):
        self.tracking = False
        self.bbox = None
        self.tracker = None
        self.last_known_bbox = None
        self.trajectory.clear()
        self.last_positions.clear()
        self.consecutive_failures = 0
        print("Status: Reset")

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

    def extract_features(self, frame, bbox):
        """Extract ORB features from a region of interest"""
        x, y, w, h = map(int, bbox)
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return None, None
        keypoints, descriptors = self.orb.detectAndCompute(roi, None)
        return keypoints, descriptors

    def compare_features(self, frame, current_bbox):
        """Compare features between initial ROI and current bbox"""
        if self.initial_features is None or current_bbox is None:
            return False
        
        # Extract features from current bbox
        _, current_descriptors = self.extract_features(frame, current_bbox)
        
        if current_descriptors is None or self.initial_features is None:
            return False
            
        # Match features
        matches = self.feature_matcher.match(self.initial_features, current_descriptors)
        
        # Calculate similarity score
        if len(matches) > 0:
            similarity_score = sum(m.distance for m in matches) / len(matches)
            return similarity_score < 50  # Threshold for similarity
        return False

    def process_frame(self, frame):
        # Process a single frame with tracking logic
        if self.tracking and self.tracker:
            # Try to update tracker
            success, new_bbox = self.tracker.update(frame)
            
            # Check if tracking is still reliable
            if success:
                # Apply Kalman filtering for prediction
                if self.kalman_filter is not None:
                    # Update Kalman with new measurement
                    self.update_kalman(new_bbox)
                    # Get prediction
                    predicted_bbox = self.predict_position()
                    
                    # Use prediction or actual depending on confidence
                    if self.consecutive_failures > 0:
                        # Blend predicted and actual positions
                        blend_factor = min(self.consecutive_failures / self.max_consecutive_failures, 0.8)
                        x1, y1, w1, h1 = predicted_bbox
                        x2, y2, w2, h2 = new_bbox
                        blended_bbox = (
                            int(x1 * blend_factor + x2 * (1 - blend_factor)),
                            int(y1 * blend_factor + y2 * (1 - blend_factor)),
                            w2, h2
                        )
                        new_bbox = blended_bbox
                    
                # Apply smoothing to reduce jitter
                new_bbox = self.apply_smoothing(new_bbox)
                
                x, y, w, h = map(int, new_bbox)
                self.last_known_bbox = (x, y, w, h)
                self.consecutive_failures = 0
                
                # Draw rectangle and text
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Tracking", (x, y - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Calculate and store center point for trajectory
                if self.show_trajectory:
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
                
                if self.consecutive_failures < self.max_consecutive_failures:
                    # Try to recover using Kalman prediction
                    if self.kalman_filter is not None and self.last_known_bbox is not None:
                        predicted_bbox = self.predict_position()
                        x, y, w, h = map(int, predicted_bbox)
                        
                        # Draw rectangle with different color to indicate prediction
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                        cv2.putText(frame, "Predicted", (x, y - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                        
                        print(f"Status: Recovering ({self.consecutive_failures})")
                else:
                    # Too many consecutive failures, reset tracker
                    if self.last_known_bbox is not None:
                        # Check feature similarity before reinitializing
                        if self.compare_features(frame, self.last_known_bbox):
                            self.bbox = self.last_known_bbox
                            self.init_tracker(frame)
                            print("Status: Reinitialized - Features matched")
                        else:
                            print("Status: Lost Track - Features don't match")
                            self.reset()
                    else:
                        print("Status: Lost Track")
        
        return frame

    def run(self):
        print("Object Tracking Console Started")
        print("Press 's' to select ROI")
        print("Press 'q' to quit")
        print("Press 'r' to reset tracking")
        
        
        while True:
            start_time = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            self.last_frame = frame.copy()
            self.frame_count += 1
            
            # Process every nth frame based on frame_skip
            if self.frame_count % (self.frame_skip + 1) == 0:
                frame = self.process_frame(frame)
            
            # Calculate and display FPS
            end_time = time.time()
            process_time = end_time - start_time
            self.process_times.append(process_time)
            
            if len(self.process_times) > 0:
                avg_process_time = sum(self.process_times) / len(self.process_times)
                self.fps = 1.0 / avg_process_time if avg_process_time > 0 else 0
                cv2.putText(frame, f"Press: 'S' to select ROI 'R' to reset tracking 'Q' to quit ", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the resulting frame
            cv2.imshow('Object Tracking', frame)

            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.select_roi()
            elif key == ord('r'):
                self.reset()
        
        # Clean up
        self.cap.release()
        #cv2.destroyAllWindows()
        print("Application closed")

def main(tracker_type="CSRT", show_trajectory=True, use_kalman=True, auto_recovery=True, trail_length=30, recovery_attempts=3):
    parser = argparse.ArgumentParser(description='Object Tracking Console')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index (default: 0)')
    parser.add_argument('--tracker', type=str, default=tracker_type,
                        choices=['CSRT', 'KCF', 'MOSSE', 'MedianFlow', 'BOOSTING'],
                        help=f'Tracker type (default: {tracker_type})')
    parser.add_argument('--trajectory', type=bool, default=show_trajectory,
                        help=f'Show trajectory (default: {show_trajectory})')
    parser.add_argument('--kalman', type=bool, default=use_kalman,
                        help=f'Use Kalman filter (default: {use_kalman})')
    parser.add_argument('--auto-recovery', type=bool, default=auto_recovery,
                        help=f'Enable auto recovery (default: {auto_recovery})')
    parser.add_argument('--max-points', type=int, default=60,
                        help='Maximum trajectory points (default: 60)')
    parser.add_argument('--frame-skip', type=int, default=0,
                        help='Process every nth frame (default: 0, process all frames)')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    app = ObjectTrackingConsole(
        camera_index=args.camera,
        tracker_type=args.tracker,
        show_trajectory=args.trajectory,
        max_points=args.max_points,
        frame_skip=args.frame_skip,
        confidence_threshold=args.confidence
    )
    
    app.run()

if __name__ == "__main__":
    main()