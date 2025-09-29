#!/bin/python3

import torch
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from dgei_interfaces.msg import GazeFrame, GazeDetection
import cv2
from cv_bridge import CvBridge
from l2cs.gaze_detectors import Gaze_Detector
import time
import threading
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

class KalmanBoxTracker:
    """
    Kalman filter for tracking bounding boxes in 2D space
    State vector: [center_x, center_y, width, height, dx, dy, dw, dh]
    """
    count = 0
    
    def __init__(self, bbox, gaze_data=None, gaze_smoothing=0.7, 
                 initial_hits=1, initial_hit_streak=1, initial_age=1):
        """Initialize Kalman filter for bounding box tracking"""
        KalmanBoxTracker.count += 1
        # Reset counter to 1 when it reaches 51 (so IDs cycle from 1-50)
        if KalmanBoxTracker.count > 50:
            KalmanBoxTracker.count = 1
        self.id = KalmanBoxTracker.count
        self.time_since_update = 0
        self.hits = initial_hits
        self.hit_streak = initial_hit_streak
        self.age = initial_age
        self.gaze_smoothing = gaze_smoothing
        
        # Convert bbox [x1, y1, x2, y2] to [center_x, center_y, width, height]
        self.bbox_to_state(bbox)
        
        # Store gaze data
        self.gaze_data = gaze_data or {'pitch': 0.0, 'yaw': 0.0, 'score': 0.0}
        
        # Initialize Kalman filter
        self.kf = self.init_kalman_filter()
        
    def init_kalman_filter(self):
        """Initialize simple Kalman filter (custom implementation)"""
        # State: [center_x, center_y, width, height, dx, dy, dw, dh]
        self.x = np.zeros(8)  # State vector
        self.x[:4] = self.state_vec
        
        # State covariance matrix
        self.P = np.eye(8) * 1000.0
        self.P[4:, 4:] *= 0.01  # Lower uncertainty for velocities
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Process noise covariance
        self.Q = np.eye(8) * 0.1
        self.Q[4:, 4:] *= 0.01
        
        # Measurement matrix
        self.H = np.eye(4, 8)
        
        # Measurement noise covariance
        self.R = np.eye(4) * 10.0
    
    def bbox_to_state(self, bbox):
        """Convert bbox [x1, y1, x2, y2] to state [center_x, center_y, width, height]"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.0
        y = bbox[1] + h/2.0
        self.state_vec = np.array([x, y, w, h])
        
    def state_to_bbox(self):
        """Convert state [center_x, center_y, width, height] to bbox [x1, y1, x2, y2]"""
        w = self.x[2]
        h = self.x[3]
        x1 = self.x[0] - w/2.0
        y1 = self.x[1] - h/2.0
        x2 = self.x[0] + w/2.0
        y2 = self.x[1] + h/2.0
        return np.array([x1, y1, x2, y2])
    
    def predict(self):
        """Predict the next state using Kalman filter"""
        # Predict step
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.state_to_bbox()
    
    def update(self, bbox, gaze_data=None):
        """Update the Kalman filter with a new detection"""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        
        # Convert bbox to measurement
        self.bbox_to_state(bbox)
        z = self.state_vec
        
        # Update step
        y = z - self.H @ self.x  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        
        # Update gaze data with configurable smoothing
        if gaze_data:
            alpha = self.gaze_smoothing  # Use configurable smoothing factor
            self.gaze_data['pitch'] = alpha * gaze_data['pitch'] + (1-alpha) * self.gaze_data['pitch']
            self.gaze_data['yaw'] = alpha * gaze_data['yaw'] + (1-alpha) * self.gaze_data['yaw']
            self.gaze_data['score'] = gaze_data['score']  # Use latest score
    
    def get_state(self):
        """Get current bounding box and gaze data"""
        return self.state_to_bbox(), self.gaze_data

class GazeTracker:
    """Multi-object tracker for gaze detections using Kalman filters"""
    
    def __init__(self, max_disappeared=20, max_distance=80, min_hits=3, gaze_smoothing=0.7, 
                 initial_hits=1, initial_hit_streak=1, initial_age=1):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.min_hits = min_hits
        self.gaze_smoothing = gaze_smoothing
        self.initial_hits = initial_hits
        self.initial_hit_streak = initial_hit_streak
        self.initial_age = initial_age
        self.trackers = []
        
    def update(self, detections):
        """
        Update tracker with new detections
        detections: list of {'bbox': [x1,y1,x2,y2], 'pitch': float, 'yaw': float, 'score': float}
        """
        # Predict all existing trackers
        predicted_bboxes = []
        for tracker in self.trackers:
            predicted_bboxes.append(tracker.predict())
        
        # If no detections, return existing trackers
        if len(detections) == 0:
            # Remove old trackers
            self.trackers = [t for t in self.trackers if t.time_since_update < self.max_disappeared]
            return self.get_current_tracks()
        
        # If no existing trackers, create new ones
        if len(self.trackers) == 0:
            for det in detections:
                self.trackers.append(KalmanBoxTracker(det['bbox'], {
                    'pitch': det['pitch'],
                    'yaw': det['yaw'], 
                    'score': det['score']
                }, self.gaze_smoothing, self.initial_hits, self.initial_hit_streak, self.initial_age))
            return self.get_current_tracks()
        
        # Associate detections to trackers using Hungarian algorithm
        detection_centers = np.array([[
            (det['bbox'][0] + det['bbox'][2])/2,
            (det['bbox'][1] + det['bbox'][3])/2
        ] for det in detections])
        
        tracker_centers = np.array([[
            (bbox[0] + bbox[2])/2,
            (bbox[1] + bbox[3])/2
        ] for bbox in predicted_bboxes])
        
        # Calculate distance matrix
        if len(tracker_centers) > 0 and len(detection_centers) > 0:
            distance_matrix = cdist(detection_centers, tracker_centers)
            
            # Use Hungarian algorithm for optimal assignment
            det_indices, track_indices = linear_sum_assignment(distance_matrix)
            
            # Filter out assignments with distance > max_distance
            valid_assignments = []
            for det_idx, track_idx in zip(det_indices, track_indices):
                if distance_matrix[det_idx, track_idx] <= self.max_distance:
                    valid_assignments.append((det_idx, track_idx))
            
            # Update matched trackers
            unmatched_detections = set(range(len(detections)))
            unmatched_trackers = set(range(len(self.trackers)))
            
            for det_idx, track_idx in valid_assignments:
                self.trackers[track_idx].update(detections[det_idx]['bbox'], {
                    'pitch': detections[det_idx]['pitch'],
                    'yaw': detections[det_idx]['yaw'],
                    'score': detections[det_idx]['score']
                })
                unmatched_detections.discard(det_idx)
                unmatched_trackers.discard(track_idx)
            
            # Create new trackers for unmatched detections
            for det_idx in unmatched_detections:
                det = detections[det_idx]
                self.trackers.append(KalmanBoxTracker(det['bbox'], {
                    'pitch': det['pitch'],
                    'yaw': det['yaw'],
                    'score': det['score']
                }, self.gaze_smoothing, self.initial_hits, self.initial_hit_streak, self.initial_age))
        
        # Remove old trackers
        self.trackers = [t for t in self.trackers if t.time_since_update < self.max_disappeared]
        
        return self.get_current_tracks()
    
    def get_current_tracks(self):
        """Get current tracks with persistent IDs"""
        tracks = []
        for tracker in self.trackers:
            if tracker.time_since_update < 1 and tracker.hits >= self.min_hits:  # Use configurable min hits
                bbox, gaze_data = tracker.get_state()
                tracks.append({
                    'id': tracker.id,
                    'bbox': bbox,
                    'pitch': gaze_data['pitch'],
                    'yaw': gaze_data['yaw'],
                    'score': gaze_data['score']
                })
        return tracks

## Extra visualisation function for id tracking here
def get_gaze_messages_and_vis_render_msg(tracked_objects, rgb_frame):
    """
    Convert tracked gaze objects and rendered frame to ROS messages
    specifically, the tracked data is converted to a GazeFrame message containing GazeDetection array
    and the rgb_frame is the image with bounding boxes and gaze directions drawn on it

    ---GazeFrame message structure:
    std_msgs/Header header
    GazeDetection[] gazes

    ---GazeDetection message structure:
    std_msgs/Header header
    uint8 id
    float64 yaw
    float64 pitch
    """
    if rgb_frame is None or not tracked_objects:
        return None, None

    # Create a copy of the frame for rendering
    rendered_frame = rgb_frame.copy()
    
    # Create the main GazeFrame message
    gaze_frame_msg = GazeFrame()
    gaze_frame_msg.header.stamp = rclpy.time.Time().to_msg()
    gaze_frame_msg.header.frame_id = "camera_frame"
    
    # Create individual GazeDetection messages for each tracked object
    gaze_detections = []
    
    # Use different colors for different IDs
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for obj in tracked_objects:
        # Create GazeDetection message
        gaze_detection = GazeDetection()
        gaze_detection.header.stamp = gaze_frame_msg.header.stamp
        gaze_detection.header.frame_id = "camera_frame"
        gaze_detection.id = obj['id']  # Use persistent tracked ID
        gaze_detection.pitch = float(obj['pitch'])
        gaze_detection.yaw = float(obj['yaw'])
        
        gaze_detections.append(gaze_detection)
        
        # Render the gaze detection on the RGB frame
        bbox = obj['bbox']
        x_min = int(bbox[0])
        y_min = int(bbox[1])
        x_max = int(bbox[2])
        y_max = int(bbox[3])
        
        # Use different color for each ID
        color = colors[(obj['id'] - 1) % len(colors)]
        
        # Draw bounding box with ID-specific color
        cv2.rectangle(rendered_frame, (x_min, y_min), (x_max, y_max), color, 3)
        
        # Draw gaze direction arrow
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        
        # Calculate arrow endpoint based on pitch and yaw (matching original draw_gaze function)
        import math
        arrow_length = 60
        pitch = obj['pitch']
        yaw = obj['yaw']
        
        # Match the original draw_gaze calculation:
        # dx = -length * sin(pitch) * cos(yaw)
        # dy = -length * sin(yaw)
        dx = -arrow_length * math.sin(pitch) * math.cos(yaw)
        dy = -arrow_length * math.sin(yaw)
        
        end_x = int(center_x + dx)
        end_y = int(center_y + dy)
        
        cv2.arrowedLine(rendered_frame, (center_x, center_y), (end_x, end_y), color, 4)
        
        # Add text labels with ID-specific background
        label_text = f"ID:{obj['id']} P:{obj['pitch']:.1f} Y:{obj['yaw']:.1f}"
        
        # Draw background rectangle for text
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(rendered_frame, (x_min, y_min - text_height - 10), 
                     (x_min + text_width, y_min), color, -1)
        
        # Draw text
        cv2.putText(rendered_frame, label_text, (x_min, y_min - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Add the detections to the main message
    gaze_frame_msg.gazes = gaze_detections
    
    return gaze_frame_msg, rendered_frame

class GazeTrackingNode(Node):
    def __init__(self):
        super().__init__('gaze_tracking_node')
        
        # Declare parameters with default values
        self.declare_parameter('camera_number', 0)  # Default to first camera
        self.declare_parameter('visualisation_topic', '/gaze/vis_frame') 
        self.declare_parameter('data_publishing_topic', '/gaze/data')
        self.declare_parameter('raw_image_topic', '/gaze/image_raw')  # Raw image topic
        self.declare_parameter('model_path', '/home/vscode/dev/gaze_ws/src/ros2_dgei/l2cs_net/weights/L2CSNet_gaze360.pkl')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('backbone_architecture', 'ResNet50')
        self.declare_parameter('publish_rate', 25.0)  # Default 25 FPS publishing
        
        # Kalman filter tracking parameters
        self.declare_parameter('tracker_max_disappeared', 20)  # Frames to keep track without detection
        self.declare_parameter('tracker_max_distance', 80.0)   # Maximum pixel distance for association
        self.declare_parameter('tracker_min_hits', 3)          # Minimum hits before showing track
        self.declare_parameter('tracker_gaze_smoothing', 0.7)  # Gaze smoothing factor (0-1)
        
        # Tracker initialization parameters
        self.declare_parameter('tracker_initial_hits', 1)       # Initial hit count for new trackers
        self.declare_parameter('tracker_initial_hit_streak', 1) # Initial hit streak for new trackers
        self.declare_parameter('tracker_initial_age', 1)        # Initial age for new trackers

        # Get parameter values
        self.camera_number = self.get_parameter('camera_number').get_parameter_value().integer_value
        self.visualisation_topic = self.get_parameter('visualisation_topic').get_parameter_value().string_value
        self.data_publishing_topic = self.get_parameter('data_publishing_topic').get_parameter_value().string_value
        self.raw_image_topic = self.get_parameter('raw_image_topic').get_parameter_value().string_value
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.backbone_architecture = self.get_parameter('backbone_architecture').get_parameter_value().string_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        
        # Get tracking parameters
        self.tracker_max_disappeared = self.get_parameter('tracker_max_disappeared').get_parameter_value().integer_value
        self.tracker_max_distance = self.get_parameter('tracker_max_distance').get_parameter_value().double_value
        self.tracker_min_hits = self.get_parameter('tracker_min_hits').get_parameter_value().integer_value
        self.tracker_gaze_smoothing = self.get_parameter('tracker_gaze_smoothing').get_parameter_value().double_value
        
        # Get tracker initialization parameters
        self.tracker_initial_hits = self.get_parameter('tracker_initial_hits').get_parameter_value().integer_value
        self.tracker_initial_hit_streak = self.get_parameter('tracker_initial_hit_streak').get_parameter_value().integer_value
        self.tracker_initial_age = self.get_parameter('tracker_initial_age').get_parameter_value().integer_value
        
        # Log the parameters
        self.get_logger().info(f'Camera number: {self.camera_number}')
        self.get_logger().info(f'Visualization topic: {self.visualisation_topic}')
        self.get_logger().info(f'Data topic: {self.data_publishing_topic}')
        self.get_logger().info(f'Raw image topic: {self.raw_image_topic}')
        self.get_logger().info(f'Model path: {self.model_path}')
        self.get_logger().info(f'Device: {self.device}')
        self.get_logger().info(f'Backbone architecture: {self.backbone_architecture}')
        self.get_logger().info(f'Publish rate: {self.publish_rate} Hz')
        self.get_logger().info(f'Tracker max disappeared: {self.tracker_max_disappeared} frames')
        self.get_logger().info(f'Tracker max distance: {self.tracker_max_distance} pixels')
        self.get_logger().info(f'Tracker min hits: {self.tracker_min_hits}')
        self.get_logger().info(f'Gaze smoothing factor: {self.tracker_gaze_smoothing}')
        self.get_logger().info(f'Tracker initial hits: {self.tracker_initial_hits}')
        self.get_logger().info(f'Tracker initial hit streak: {self.tracker_initial_hit_streak}')
        self.get_logger().info(f'Tracker initial age: {self.tracker_initial_age}')
        
        # Thread-safe variables for sharing data between threads
        self.frame_lock = threading.Lock()
        self.latest_vis_frame = None
        self.latest_raw_frame = None  # Store the raw frame before processing
        self.latest_gaze_msg = None  # Store the processed GazeFrame message
        self.running = True
        
        # Initialize Kalman filter tracker with ROS parameters
        self.tracker = GazeTracker(
            max_disappeared=self.tracker_max_disappeared,
            max_distance=self.tracker_max_distance,
            min_hits=self.tracker_min_hits,
            gaze_smoothing=self.tracker_gaze_smoothing,
            initial_hits=self.tracker_initial_hits,
            initial_hit_streak=self.tracker_initial_hit_streak,
            initial_age=self.tracker_initial_age
        )
        
        # Initialize gaze tracking components
        self.setup_gaze_tracking()
        
        # Start background gaze processing thread
        self.gaze_thread = threading.Thread(target=self.gaze_processing_loop, daemon=True)
        self.gaze_thread.start()
        
        # Create timer for publishing at fixed rate (25Hz)
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_callback)
        
    def setup_gaze_tracking(self):
        """Initialize gaze tracking model and publishers"""
        self.get_logger().info("Initializing gaze detector...")
        self.Gaze_detector = Gaze_Detector(
            device=self.device,
            nn_arch=self.backbone_architecture,
            weights_pth=self.model_path
        )

        # Initialize publishers
        self.rgb_publisher = self.create_publisher(Image, self.visualisation_topic, 10) 
        self.raw_image_publisher = self.create_publisher(Image, self.raw_image_topic, 10)  # Raw image publisher
        self.gaze_data_publisher = self.create_publisher(GazeFrame, self.data_publishing_topic, 10)

        # Initialize camera
        self.get_logger().info(f"Opening camera {self.camera_number}...")
        self.cap = cv2.VideoCapture(int(self.camera_number))
        if not self.cap.isOpened():
            self.get_logger().error("Cannot open webcam")
            raise RuntimeError("Cannot open webcam")
        
        # Initialize cv_bridge
        self.bridge = CvBridge()
        self.get_logger().info("Gaze tracking setup complete!")
    
    def cleanup(self):
        """Clean up resources"""
        self.get_logger().info("Starting cleanup...")
        self.running = False
        
        # Wait for gaze thread to finish
        if hasattr(self, 'gaze_thread') and self.gaze_thread.is_alive():
            self.get_logger().info("Waiting for gaze processing thread to finish...")
            self.gaze_thread.join(timeout=2.0)
        
        # Release camera
        if hasattr(self, 'cap') and self.cap is not None:
            self.get_logger().info("Releasing camera capture...")
            self.cap.release()
            self.cap = None
        
        self.get_logger().info("Cleanup complete!")
    
    def gaze_processing_loop(self):
        """Continuous gaze processing in background thread"""
        self.get_logger().info("Starting gaze processing thread...")
        
        with torch.no_grad():
            while self.running:
                try:
                    # Get frame from camera
                    success, frame = self.cap.read()
                    
                    if not success:
                        self.get_logger().warn("Failed to read frame from camera")
                        continue

                    # Store raw frame (make a copy to avoid modifications affecting raw frame)
                    raw_frame = frame.copy()

                    # Process gaze detection
                    g_success = self.Gaze_detector.detect_gaze(frame)
                    
                    # Get visualization frame
                    
                    # Get gaze data and process visualization
                    gaze_msg = None
                    vframe = None
                    
                    # Prepare detections for tracking
                    detections = []
                    base_frame = frame  # Use original frame for rendering
                    
                    if g_success:
                        try:
                            results = self.Gaze_detector.get_latest_gaze_results()
                            
                            # Convert L2CS results to detection format for tracker
                            for bbox, pitch, yaw, score in zip(results.bboxes, results.pitch, results.yaw, results.scores):
                                detections.append({
                                    'bbox': bbox,
                                    'pitch': pitch,
                                    'yaw': yaw,
                                    'score': score
                                })
                                
                        except Exception as e:
                            self.get_logger().debug(f"Could not process gaze results: {e}")
                    
                    # Update tracker with detections (empty list if no detections)
                    tracked_objects = self.tracker.update(detections)
                    
                    # Create ROS messages and visualization
                    if tracked_objects:
                        gaze_msg, vframe = get_gaze_messages_and_vis_render_msg(tracked_objects, base_frame)
                    else:
                        gaze_msg = None
                        vframe = base_frame
                    
                    # Thread-safe update of shared data
                    with self.frame_lock:
                        self.latest_vis_frame = vframe
                        self.latest_raw_frame = raw_frame  # Store the raw frame
                        self.latest_gaze_msg = gaze_msg
                        
                except Exception as e:
                    self.get_logger().error(f"Error in gaze processing loop: {e}")
                    time.sleep(0.1)  # Pause on error
                    
        self.get_logger().info("Gaze processing thread stopped")
    
    def publish_callback(self):
        """Publisher callback - runs at fixed 25Hz rate"""
        try:
            # Get latest processed data (thread-safe)
            with self.frame_lock:
                vis_frame = self.latest_vis_frame
                raw_frame = self.latest_raw_frame
                gaze_msg = self.latest_gaze_msg
            
            # Publish gaze data message if available
            if gaze_msg is not None:
                self.gaze_data_publisher.publish(gaze_msg)
                
                # Log detection info occasionally
                if hasattr(self, '_last_log_time'):
                    if time.time() - self._last_log_time > 2.0:  # Log every 2 seconds
                        self.get_logger().info(f"Published gaze data for {len(gaze_msg.gazes)} faces")
                        self._last_log_time = time.time()
                else:
                    self._last_log_time = time.time()
            
            # Publish visualization frame if available
            if vis_frame is not None and isinstance(vis_frame, np.ndarray):
                ros_image = self.bridge.cv2_to_imgmsg(vis_frame, encoding="bgr8")
                self.rgb_publisher.publish(ros_image)
            
            # Publish raw image frame if available
            if raw_frame is not None and isinstance(raw_frame, np.ndarray):
                raw_ros_image = self.bridge.cv2_to_imgmsg(raw_frame, encoding="bgr8")
                self.raw_image_publisher.publish(raw_ros_image)
                    
        except Exception as e:
            self.get_logger().error(f"Error in publish callback: {e}")


def main(args=None):
    rclpy.init(args=args)
    
    gaze_node = None
    try:
        gaze_node = GazeTrackingNode()
        rclpy.spin(gaze_node)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, shutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if gaze_node is not None:
            gaze_node.cleanup()  # Ensure camera is released
            gaze_node.destroy_node()
        rclpy.shutdown()
        print("Shutdown complete.")


if __name__ == '__main__':
    main()