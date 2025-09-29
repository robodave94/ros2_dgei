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

## Extra visualisation function for id tracking here
def get_gaze_messages_and_vis_render_msg(gaze_data, rgb_frame):
    """
    Convert the gaze data and rendered frame to ROS messages
    specifically, the gaze data is converted to a GazeFrame message containing GazeDetection array
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
    if rgb_frame is None or gaze_data is None:
        return None, None

    # Create a copy of the frame for rendering
    rendered_frame = rgb_frame.copy()
    
    # Create the main GazeFrame message
    gaze_frame_msg = GazeFrame()
    gaze_frame_msg.header.stamp = rclpy.time.Time().to_msg()
    gaze_frame_msg.header.frame_id = "camera_frame"
    
    # Create individual GazeDetection messages for each detected face
    gaze_detections = []
    
    for i, (bbox, pitch, yaw, score) in enumerate(zip(gaze_data['bboxes'], gaze_data['pitch'], gaze_data['yaw'], gaze_data['scores'])):
        # Create GazeDetection message
        gaze_detection = GazeDetection()
        gaze_detection.header.stamp = gaze_frame_msg.header.stamp
        gaze_detection.header.frame_id = "camera_frame"
        gaze_detection.id = i  # Assign a unique ID to each detected face
        gaze_detection.pitch = float(pitch)
        gaze_detection.yaw = float(yaw)
        
        gaze_detections.append(gaze_detection)
        
        # Render the gaze detection on the RGB frame
        x_min = int(bbox[0])
        y_min = int(bbox[1])
        x_max = int(bbox[2])
        y_max = int(bbox[3])
        
        # Draw bounding box
        cv2.rectangle(rendered_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # # Draw gaze direction arrow (simple visualization)
        # center_x = (x_min + x_max) // 2
        # center_y = (y_min + y_max) // 2
        
        # # Calculate arrow endpoint based on pitch and yaw (simplified)
        # import math
        # arrow_length = 50
        # end_x = int(center_x + arrow_length * math.sin(yaw))
        # end_y = int(center_y - arrow_length * math.sin(pitch))
        
        # cv2.arrowedLine(rendered_frame, (center_x, center_y), (end_x, end_y), (0, 0, 255), 3)
        
        # Add text labels
        label_text = f'ID: {i} | P: {pitch:.2f} | Y: {yaw:.2f} | S: {score:.2f}'
        cv2.putText(rendered_frame, label_text, (x_min, y_min - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
        self.declare_parameter('model_path', '/home/vscode/dev/gaze_ws/src/ros2_dgei/l2cs_net/weights/L2CSNet_gaze360.pkl')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('backbone_architecture', 'ResNet50')
        self.declare_parameter('publish_rate', 25.0)  # Default 25 FPS publishing

        # Get parameter values
        self.camera_number = self.get_parameter('camera_number').get_parameter_value().integer_value
        self.visualisation_topic = self.get_parameter('visualisation_topic').get_parameter_value().string_value
        self.data_publishing_topic = self.get_parameter('data_publishing_topic').get_parameter_value().string_value
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.backbone_architecture = self.get_parameter('backbone_architecture').get_parameter_value().string_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        
        # Log the parameters
        self.get_logger().info(f'Camera number: {self.camera_number}')
        self.get_logger().info(f'Visualization topic: {self.visualisation_topic}')
        self.get_logger().info(f'Data topic: {self.data_publishing_topic}')
        self.get_logger().info(f'Model path: {self.model_path}')
        self.get_logger().info(f'Device: {self.device}')
        self.get_logger().info(f'Backbone architecture: {self.backbone_architecture}')
        self.get_logger().info(f'Publish rate: {self.publish_rate} Hz')
        
        # Thread-safe variables for sharing data between threads
        self.frame_lock = threading.Lock()
        self.latest_vis_frame = None
        self.latest_gaze_msg = None  # Store the processed GazeFrame message
        self.running = True
        
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

                    # Process gaze detection
                    g_success = self.Gaze_detector.detect_gaze(frame)
                    
                    # Get visualization frame
                    
                    # Get gaze data and process visualization
                    gaze_msg = None
                    vframe = None
                    
                    if g_success:
                        try:
                            results = self.Gaze_detector.get_latest_gaze_results()
                            gaze_data = {
                                'bboxes': results.bboxes,
                                'pitch': results.pitch,
                                'yaw': results.yaw,
                                'scores': results.scores
                            }
                            
                            # Get the base visualization frame and process it
                            base_frame = self.Gaze_detector.draw_gaze_window()
                            gaze_msg, vframe = get_gaze_messages_and_vis_render_msg(gaze_data, base_frame)
                            
                        except Exception as e:
                            self.get_logger().debug(f"Could not process gaze results: {e}")
                            # Fallback to basic visualization
                            vframe = self.Gaze_detector.draw_gaze_window()
                    else:
                        # No detection, just get the basic frame
                        vframe = self.Gaze_detector.draw_gaze_window()
                    
                    # Thread-safe update of shared data
                    with self.frame_lock:
                        self.latest_vis_frame = vframe
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