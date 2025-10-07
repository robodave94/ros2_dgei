#!/bin/python3

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from dgei_interfaces.msg import GazeFrame, GazeDetection
from dgei_interfaces.srv import EntropyCalculation
from cv_bridge import CvBridge

from dgei_gaze.data import read_gaze_config, write_yaml_file

from l2cs.gaze_detectors import Gaze_Detector
from dgei_gaze.entropy import AttentionTrackerCollection, AttentionCalibrator

from threading import Thread, Lock

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

    # Calculate center of face for gaze arrow
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Draw gaze direction arrow
    arrow_length = 80
    pitch_rad = math.radians(gaze_data_dict["smoothed_pitch"])
    yaw_rad = math.radians(gaze_data_dict["smoothed_yaw"])
    
    # Calculate arrow endpoint (matching gaze direction calculation)
    dx = -arrow_length * math.sin(pitch_rad) * math.cos(yaw_rad)
    dy = -arrow_length * math.sin(yaw_rad)
    
    end_x = int(center_x + dx)
    end_y = int(center_y + dy)
    
    # Draw gaze arrow in bright yellow
    cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), (0, 255, 255), 3)
    
    # Function to interpolate between red and green based on score (0-100)
    def score_to_color(score):
        # Clamp score between 0 and 100
        score = max(0, min(100, score))
        # Interpolate from red (0,0,255) to green (0,255,0)
        red_component = int(255 * (100 - score) / 100)
        green_component = int(255 * score / 100)
        return (0, green_component, red_component)
    
    # Get colors based on scores
    gaze_score_color = score_to_color(gaze_data_dict["gaze_score"])
    # For entropy, lower is better, so invert it (higher entropy = lower score)
    entropy_score = max(0, 100 - (gaze_data_dict["gaze_entropy"] * 50))  # Scale entropy to 0-100 range
    entropy_color = score_to_color(entropy_score)
    
    # Attention colors
    attention_color = (0, 255, 0) if gaze_data_dict["attention_state"] else (0, 0, 255)
    sustained_color = (0, 255, 0) if gaze_data_dict["sustained_attention"] else (0, 0, 255)

    # Draw bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
    
    # Draw text with appropriate colors
    cv2.putText(frame, f'ID:{id}', (x, y - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f'GazeScore:{gaze_data_dict["gaze_score"]:.1f}', (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gaze_score_color, 1)
    cv2.putText(frame, f'Entropy:{gaze_data_dict["gaze_entropy"]:.2f}', (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, entropy_color, 1)
    cv2.putText(frame, f'RobotLooks:{gaze_data_dict["robot_looks"]}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)

    # Attention states with green/red colors
    attention_text = "ATTENTION" if gaze_data_dict["attention_state"] else "No Attention"
    cv2.putText(frame, attention_text, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, attention_color, 1)

    sustained_text = "SUSTAINED" if gaze_data_dict["sustained_attention"] else "Not Sustained"
    cv2.putText(frame, sustained_text, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, sustained_color, 1)

    # Place pitch and yaw on one line
    cv2.putText(frame, f'P:{gaze_data_dict["smoothed_pitch"]:.1f} Y:{gaze_data_dict["smoothed_yaw"]:.1f}',
                 (x, y + h + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame

class EntropyGazeNode(Node):
    def __init__(self):
        super().__init__('entropy_based_gaze_tracking')
        
        # Initialize CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Declare ROS parameters with default values
        self.declare_parameter('image_topic', '/gaze/image_raw')
        self.declare_parameter('gaze_topic', '/gaze/data')
        self.declare_parameter('service_name', 'entropy_calculation')
        self.declare_parameter('entropy_viz_topic', '/entropy_visualization')
        self.declare_parameter('attention_configuration', '/home/vscode/dev/gaze_ws/src/ros2_dgei/dgei_gaze/config/data.yaml')
        self.declare_parameter('sync_time_tolerance', 0.05)  # 50ms tolerance for synchronization
        
        # Get parameter values
        self.sync_time_tolerance = self.get_parameter('sync_time_tolerance').get_parameter_value().double_value
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        gaze_topic = self.get_parameter('gaze_topic').get_parameter_value().string_value
        service_name = self.get_parameter('service_name').get_parameter_value().string_value
        entropy_viz_topic = self.get_parameter('entropy_viz_topic').get_parameter_value().string_value
        self.attention_config_path = self.get_parameter('attention_configuration').get_parameter_value().string_value

        self.attention_config = read_gaze_config(self.attention_config_path)

        # Initialize synchronization variables
        self.image_buffer = deque(maxlen=1)  # Store recent images with timestamps
        self.gaze_buffer = deque(maxlen=1)   # Store recent gaze data with timestamps
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
            

        # Calibration state
        self.call_calibration = False
        # Calibration get/set lock for multi-threading
        self.calibration_lock = Lock()

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
        
        # Create service using parameter value
        self.entropy_gaze_calibration_service = self.create_service(
            EntropyCalculation,
            service_name,
            self.gaze_calibration_callback
        )
        
        self.get_logger().info('Entropy Gaze tracking node initialized')
        self.get_logger().info(f'Subscribed to: {image_topic} and {gaze_topic}')
        self.get_logger().info(f'Service available: {service_name}')
        self.get_logger().info(f'Publishing entropy visualization to: {entropy_viz_topic}')
        self.get_logger().info(f'Synchronization tolerance: {self.sync_time_tolerance}s')
        self.get_logger().info(f'Loaded attention config: {self.attention_config}')

        # Start a separate thread to process synchronized data
        self.processing_thread = Thread(target=self.process_gaze_data_into_entropy_attention)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    
    def get_is_calibrating(self):
        """Thread-safe getter for calibration state"""
        with self.calibration_lock:
            return self.call_calibration
    
    def set_is_calibrating(self, state: bool):
        """Thread-safe setter for calibration state"""
        with self.calibration_lock:
            self.call_calibration = state

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
        """Attempt to synchronize image and gaze data based on timestamps, prioritizing latest messages"""
        if not self.image_buffer or not self.gaze_buffer:
            return
        
        # Get the latest (most recent) messages from each buffer
        latest_img_timestamp, latest_img_cv, latest_img_msg = self.image_buffer[-1]
        latest_gaze_timestamp, latest_gaze_msg = self.gaze_buffer[-1]
        
        # First try to match the latest messages with each other
        latest_time_diff = abs(latest_img_timestamp - latest_gaze_timestamp)
        
        if latest_time_diff <= self.sync_time_tolerance:
            # Latest messages are synchronized, use them
            best_match = (latest_img_cv, latest_gaze_msg, latest_img_timestamp, latest_gaze_timestamp)
            min_time_diff = latest_time_diff
        else:
            # Latest messages don't sync well, search for best match prioritizing recent data
            best_match = None
            min_time_diff = float('inf')
            
            # Search backwards through buffers (newest to oldest) to prioritize recent matches
            for img_timestamp, img_cv, img_msg in reversed(self.image_buffer):
                for gaze_timestamp, gaze_msg in reversed(self.gaze_buffer):
                    time_diff = abs(img_timestamp - gaze_timestamp)
                    
                    if time_diff <= self.sync_time_tolerance:
                        # Prioritize matches that are more recent
                        # Use a composite score: time_diff + age_penalty
                        age_penalty_img = latest_img_timestamp - img_timestamp
                        age_penalty_gaze = latest_gaze_timestamp - gaze_timestamp
                        total_age_penalty = (age_penalty_img + age_penalty_gaze) * 0.1  # Weight age penalty
                        
                        composite_score = time_diff + total_age_penalty
                        
                        if composite_score < min_time_diff:
                            min_time_diff = time_diff  # Store actual time diff, not composite score
                            best_match = (img_cv, gaze_msg, img_timestamp, gaze_timestamp)
                        
                        # If we find a very good recent match, prefer it over searching further
                        if time_diff < 0.01 and (latest_img_timestamp - img_timestamp) < 0.1:
                            break
                
                # Early exit if we found a very good match
                if best_match and min_time_diff < 0.01:
                    break

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

    def gaze_calibration_callback(self, request, response):
        """Service callback for gaze calibration requests"""
        # Set calibration state
        self.set_is_calibrating(True)
        
        calibration_time = request.calibration_time
        frames_required = request.frames_required
        angle_tolerance = request.angle_tolerance

        # Initialise calibrator
        calibrator = AttentionCalibrator(   calibration_time=calibration_time,
                                            samples_needed=frames_required,
                                            angle_tolerance=angle_tolerance)
        
        calibrator.start_calibration()
        
        while self.get_is_calibrating() and rclpy.ok():
            sync_image, sync_gaze = self.get_synchronized_data()
            if sync_image is None or sync_gaze is None:
                self.get_logger().info('Waiting for synchronized data to start calibration...')
                sleep(0.1)
                continue
            
            face_detections = []
            for gaze_det in sync_gaze.gazes:
                # Process each gaze detection
                # Get face ID and gaze angles
                face_id = gaze_det.id
                pitch = math.degrees(gaze_det.pitch)
                yaw = math.degrees(gaze_det.yaw)

                # setup
                face_detections.append({'id': face_id, 'pitch': pitch, 'yaw': yaw})

            if len(face_detections) == 1:
                # get the pitch and yaw of the single face
                pitch = face_detections[0]['pitch']
                yaw = face_detections[0]['yaw']
                status, msg = calibrator.process_calibration_frame(pitch, yaw)
                if status:
                    self.get_logger().info('Calibration successful')
                    self.set_is_calibrating(False)
                    # print the message
                    self.get_logger().info('----')
                    self.get_logger().info(msg)
                    self.get_logger().info('----')
                    break
                else:
                    self.get_logger().info(msg)
            else:
                self.get_logger().info('Calibration requires exactly one face in the frame')
                # Calibration cannot proceed without exactly one face, unsuccessul attempt and cancel operation
                response.success = False
                self.set_is_calibrating(False)
                return response
            

            sleep(0.02)

        if calibrator.is_calibrated:
                self.get_logger().info('Calibration complete, computing thresholds...')

        self.attention_tracker_collection.update_calibration_dictionary(calibrator.get_calibrated_parameters())
        write_yaml_file(calibrator.get_calibrated_parameters(),self.attention_config_path)

        self.get_logger().info('Calibration parameters updated in Attention Tracker Collection')
        
        response.success = True
        self.set_is_calibrating(False)

        return response


    
    ## Function to constantly process synchronized data gaze calculations
    def process_gaze_data_into_entropy_attention(self):
        """Process synchronized data to compute entropy-based attention"""
        # First delay the processing loop by 5 seconds to allow buffers to fill
        sleep(7)

        sync_image, sync_gaze = self.get_synchronized_data()
        last_header_stamp = sync_gaze.header.stamp
        while rclpy.ok():
            # get is_calibrating state
            is_calibrating = self.get_is_calibrating()
            # if calibrating, skip processing
            if is_calibrating:
                self.get_logger().info('Calibration in progress, skipping processing')
                sleep(0.1)
                continue

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


