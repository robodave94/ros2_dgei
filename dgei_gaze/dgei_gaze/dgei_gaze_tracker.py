#!/bin/python3

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from dgei_interfaces.msg import GazeFrame, GazeDetection
from dgei_interfaces.srv import EntropyCalculation
from cv_bridge import CvBridge

from dgei_gaze.data import read_gaze_config

from l2cs.gaze_detectors import Gaze_Detector
from dgei_gaze.entropy import AttentionTracker, AttentionTrackerCollection

from threading import Thread

import math
import pdb
import cv2
import numpy as np
from time import time, sleep
from collections import deque

def visualise_attention_entropy_on_frame(id, gaze_data_dict, gaze_msg, frame):
    """
    Visualize attention and entropy information on the given frame.

    Gaze dict:
    'attention_state': attention,
    'sustained_attention': sustained_attention,
    'smoothed_pitch': smoothed_pitch,
    'smoothed_yaw': smoothed_yaw,
    'original_pitch': pitch,
    'original_yaw': yaw,
    'gaze_score': metrics['gaze_score'],
    'robot_looks': metrics['robot_looks'],
    'gaze_entropy': metrics['gaze_entropy']

    gaze_msg:
    bounding_box_x 
    bounding_box_y 
    bounding_box_width 
    bounding_box_height


    Args:
        id (int): The ID of the face/gaze detection.
        gaze_data_dict (dict): Dictionary containing gaze data with keys 'gaze_score' and 'gaze_entropy'.
        gaze_msg (GazeFrame): A single gaze detection message.
        frame (np.ndarray): The image frame to draw on.

        
        
    Draws on frames:
        - Bounding boxes around detected faces
        - Gaze score and entropy information
        - robot looks

    Returns:
        np.ndarray: The image frame with visualizations.
    """
    # Draw bounding box
    x = int(gaze_msg.bounding_box_x)
    y = int(gaze_msg.bounding_box_y)
    w = int(gaze_msg.bounding_box_width)
    h = int(gaze_msg.bounding_box_height)

    # print(f"X:{x} Y:{y} W:{w} H:{h}")

    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.putText(frame, f'ID:{id}', (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(frame, f'GazeScore:{gaze_data_dict["gaze_score"]:.1f}', (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(frame, f'GazeEntropy:{gaze_data_dict["gaze_entropy"]:.1f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    # cv2.putText(frame, f'RobotLooks:{gaze_data_dict["robot_looks"]}', (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    # cv2.putText(frame, f'Attention:{gaze_data_dict["attention_state"]}', (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    # cv2.putText(frame, f'Sustained:{gaze_data_dict["sustained_attention"]}', (x, y + h + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    # # Place pitch and yaw on one line
    # cv2.putText(frame, f'Pitch:{gaze_data_dict["smoothed_pitch"]:.1f} Yaw:{gaze_data_dict["smoothed_yaw"]:.1f}',
    #              (x, y + h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return frame

class EntropyGazeNode(Node):
    def __init__(self):
        super().__init__('entropy_based_gaze_tracking')
        
        # Initialize CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Declare ROS parameters with default values
        self.declare_parameter('calibration_time', 10.0)
        self.declare_parameter('frames_required', 300)
        self.declare_parameter('deg_angle_tolerance', 15.0)
        self.declare_parameter('image_topic', '/gaze/image_raw')
        self.declare_parameter('gaze_topic', '/gaze/data')
        self.declare_parameter('service_name', 'entropy_calculation')
        self.declare_parameter('entropy_viz_topic', '/entropy_visualization')
        self.declare_parameter('attention_configuration', '/home/vscode/dev/gaze_ws/src/ros2_dgei/dgei_gaze/config/data.yaml')
        self.declare_parameter('sync_time_tolerance', 0.05)  # 50ms tolerance for synchronization
        
        # Get parameter values
        self.calibration_time = self.get_parameter('calibration_time').get_parameter_value().double_value
        self.frames_required = self.get_parameter('frames_required').get_parameter_value().integer_value
        self.deg_angle_tolerance = self.get_parameter('deg_angle_tolerance').get_parameter_value().double_value
        self.rad_angle_tolerance = math.radians(self.deg_angle_tolerance)
        self.sync_time_tolerance = self.get_parameter('sync_time_tolerance').get_parameter_value().double_value
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        gaze_topic = self.get_parameter('gaze_topic').get_parameter_value().string_value
        service_name = self.get_parameter('service_name').get_parameter_value().string_value
        entropy_viz_topic = self.get_parameter('entropy_viz_topic').get_parameter_value().string_value
        attention_config_path = self.get_parameter('attention_configuration').get_parameter_value().string_value

        self.attention_config = read_gaze_config(attention_config_path)

        # Initialize synchronization variables
        self.image_buffer = deque(maxlen=30)  # Store recent images with timestamps
        self.gaze_buffer = deque(maxlen=30)   # Store recent gaze data with timestamps
        self.latest_synchronized_pair = None  # Stores (image_cv, gaze_msg, sync_timestamp) tuple
        
        # For backward compatibility
        self.latest_image = None
        self.latest_gaze_data = None

        # Initialize Attention Tracker Collection
        self.attention_tracker_collection = AttentionTrackerCollection(
            self.attention_config,
            default_attention_threshold=0.5,
            default_history_size=10,
            auto_cleanup=True,
            cleanup_timeout=3.0)
            

        # Create subscriptions using parameter values
        self.image_subscription = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )
        
        self.gaze_subscription = self.create_subscription(
            GazeFrame,
            gaze_topic,
            self.gaze_callback,
            10
        )
        
        # Create publisher for entropy visualization
        self.entropy_viz_publisher = self.create_publisher(
            Image,
            entropy_viz_topic,
            10
        )
        
        # # Create service using parameter value
        # self.entropy_service = self.create_service(
        #     EntropyCalculation,
        #     service_name,
        #     self.entropy_calculation_callback
        # )
        
        self.get_logger().info('Entropy Gaze tracking node initialized')
        self.get_logger().info(f'Subscribed to: {image_topic} and {gaze_topic}')
        self.get_logger().info(f'Service available: {service_name}')
        self.get_logger().info(f'Publishing entropy visualization to: {entropy_viz_topic}')
        self.get_logger().info(f'Parameters - Calibration time: {self.calibration_time}s, Frames: {self.frames_required}, Angle tolerance: {self.deg_angle_tolerance}°')
        self.get_logger().info(f'Synchronization tolerance: {self.sync_time_tolerance}s')
        self.get_logger().info(f'Loaded attention config: {self.attention_config}')

        # Start a separate thread to process synchronized data
        self.processing_thread = Thread(target=self.process_gaze_data_into_entropy_attention)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    
    def image_callback(self, msg):
        """Callback for raw image data"""
        try:
            # Convert ROS Image message to OpenCV format
            image_cv = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Store in buffer with timestamp
            timestamp = self.msg_stamp_to_seconds(msg.header.stamp)
            self.image_buffer.append((timestamp, image_cv, msg))
            
            # Update latest image for backward compatibility
            self.latest_image = image_cv
            
            # Attempt synchronization
            self.attempt_synchronization()
            
            self.get_logger().debug(f'Received image frame at {timestamp:.3f}')
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
    
    def gaze_callback(self, msg):
        """Callback for gaze data"""
        # Store in buffer with timestamp
        timestamp = self.msg_stamp_to_seconds(msg.header.stamp)
        self.gaze_buffer.append((timestamp, msg))
        
        # Update latest gaze data for backward compatibility
        self.latest_gaze_data = msg
        
        # Attempt synchronization
        self.attempt_synchronization()
        
        self.get_logger().debug(f'Received gaze data at {timestamp:.3f}: {len(msg.gazes)} detections')
    
    def msg_stamp_to_seconds(self, stamp):
        """Convert ROS timestamp to seconds as float"""
        return stamp.sec + stamp.nanosec * 1e-9
    
    def attempt_synchronization(self):
        """Attempt to synchronize image and gaze data based on timestamps"""
        if not self.image_buffer or not self.gaze_buffer:
            return
        
        # Find the best matching pair
        best_match = None
        min_time_diff = float('inf')
        
        for img_timestamp, img_cv, img_msg in self.image_buffer:
            for gaze_timestamp, gaze_msg in self.gaze_buffer:
                time_diff = abs(img_timestamp - gaze_timestamp)
                
                if time_diff < min_time_diff and time_diff <= self.sync_time_tolerance:
                    min_time_diff = time_diff
                    best_match = (img_cv, gaze_msg, img_timestamp, gaze_timestamp)
                    # print(f'Found potential match: img={img_timestamp:.3f}, gaze={gaze_timestamp:.3f}, diff={time_diff:.3f}s')
                    # #print the message header stamp information
                    # print(f'Image header stamp: sec={img_msg.header.stamp.sec}, nanosec={img_msg.header.stamp.nanosec}')
                    # print(f'Gaze header stamp: sec={gaze_msg.header.stamp.sec}, nanosec={gaze_msg.header.stamp.nanosec}')   

        # If we found a good match, update synchronized pair and trigger processing
        if best_match is not None:
            img_cv, gaze_msg, img_time, gaze_time = best_match
            
            # Only update if this is a new pair (avoid duplicate processing)
            if (self.latest_synchronized_pair is None or 
                self.latest_synchronized_pair[2] != img_time or 
                len(self.latest_synchronized_pair) < 4 or
                self.latest_synchronized_pair[3] != gaze_time):
                
                self.latest_synchronized_pair = (img_cv, gaze_msg, img_time, gaze_time)
                
                # print(f'Synchronized pair: img={img_time:.3f}, gaze={gaze_time:.3f}, diff={min_time_diff:.3f}s')
                
                # Process the synchronized data
                self.process_synchronized_data(img_cv, gaze_msg)
                return
            else:
                # print('Synchronized pair is the same as the last one, skipping processing')
                return
        else:
            # print('best match is None')
            return
    
    def process_synchronized_data(self, image_cv, gaze_msg):
        """Process synchronized image and gaze data"""
        # Update the latest synchronized data for use in other methods
        temp_image = self.latest_image
        temp_gaze = self.latest_gaze_data
        
        # Set synchronized data as current
        self.latest_image = image_cv
        self.latest_gaze_data = gaze_msg
        
        # # Generate and publish entropy visualization with synchronized data
        # if hasattr(self, 'publish_entropy_visualization'):
        #     self.publish_entropy_visualization()
    
    def get_synchronized_data(self):
        """Get the latest synchronized image and gaze data pair"""
        if self.latest_synchronized_pair is not None:
            return self.latest_synchronized_pair[0], self.latest_synchronized_pair[1]
        return None, None
    
    ## Function to constantly process synchronized data gaze calculations
    def process_gaze_data_into_entropy_attention(self):
        """Process synchronized data to compute entropy-based attention"""
        # First delay the processing loop by 5 seconds to allow buffers to fill
        sleep(5)

        sync_image, sync_gaze = self.get_synchronized_data()
        last_header_stamp = sync_gaze.header.stamp
        while rclpy.ok():
            sync_image, sync_gaze = self.get_synchronized_data()

            if sync_image is None or sync_gaze is None:
                self.get_logger().warn('No synchronized data available for processing')
                self.attention_tracker_collection.manual_cleanup()
                continue
            elif last_header_stamp == sync_gaze.header.stamp:
                self.attention_tracker_collection.manual_cleanup()
                sleep(0.01)  # Sleep briefly to avoid busy-waiting
                continue
            else:
                last_header_stamp = sync_gaze.header.stamp
                # Synchronized data has already been validated in the callback, proceed to processing
                # for loop setup the dictionaries
                # face_detections: List of dicts with keys 'id', 'pitch', 'yaw'
                #            e.g. [{'id': 1, 'pitch': 10.5, 'yaw': -5.2}, ...]
                face_detections = []
                gaze_detections = {}
                for gaze_det in sync_gaze.gazes:
                    # Process each gaze detection
                    # Get face ID and gaze angles
                    face_id = gaze_det.id
                    pitch = math.degrees(gaze_det.pitch)
                    yaw = math.degrees(gaze_det.yaw)
                    gaze_detections[face_id] = gaze_det

                    # setup
                    face_detections.append({'id': face_id, 'pitch': pitch, 'yaw': yaw})

                gaze_entropy_viz = sync_image.copy()
                # Sleep briefly to avoid busy-waiting
                if len(face_detections) > 0:
                    results = self.attention_tracker_collection.process_frame_data(face_detections)
                    # Fix: iterate over the dictionary items to get both face_id and result_data
                    for face_id, result_data in results.items():
                        # self.get_logger().info(f"Face ID {face_id}: Attention={result_data['attention_state']}, Smoothed Pitch={result_data['smoothed_pitch']:.1f}, Smoothed Yaw={result_data['smoothed_yaw']:.1f}")
                        # Pass the face_id separately to the visualization function
                        gaze_entropy_viz = visualise_attention_entropy_on_frame(face_id, result_data, gaze_detections[face_id], gaze_entropy_viz)
                
                # publish the visualization
                if self.entropy_viz_publisher is not None:
                    try:
                        viz_msg = self.bridge.cv2_to_imgmsg(gaze_entropy_viz, encoding="bgr8")
                        viz_msg.header = sync_gaze.header  # Preserve original header info
                        self.entropy_viz_publisher.publish(viz_msg)
                        # self.get_logger().info('Published entropy visualization frame')
                    except Exception as e:
                        self.get_logger().error(f'Error publishing entropy visualization: {str(e)}')

                # sleep(0.02)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        gaze_node = EntropyGazeNode()
        rclpy.spin(gaze_node)
    except KeyboardInterrupt:
        pass
    finally:
        gaze_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


