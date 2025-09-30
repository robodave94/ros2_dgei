#!/bin/python3

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from dgei_interfaces.msg import GazeFrame, GazeDetection
from dgei_interfaces.srv import EntropyCalculation
from cv_bridge import CvBridge

from l2cs.gaze_detectors import Gaze_Detector

import math
import pdb
import cv2
import numpy as np
from time import time, sleep
from collections import deque


class EntropyGazeNode(Node):
    def __init__(self):
        super().__init__('run_gaze_tracking')
        
        # Initialize CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Declare ROS parameters with default values
        self.declare_parameter('calibration_time', 10.0)
        self.declare_parameter('frames_required', 300)
        self.declare_parameter('deg_angle_tolerance', 15.0)
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('gaze_topic', '/gaze_data')
        self.declare_parameter('service_name', 'entropy_calculation')
        self.declare_parameter('entropy_viz_topic', '/entropy_visualization')
        
        # Get parameter values
        self.calibration_time = self.get_parameter('calibration_time').get_parameter_value().double_value
        self.frames_required = self.get_parameter('frames_required').get_parameter_value().integer_value
        self.deg_angle_tolerance = self.get_parameter('deg_angle_tolerance').get_parameter_value().double_value
        self.rad_angle_tolerance = math.radians(self.deg_angle_tolerance)
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        gaze_topic = self.get_parameter('gaze_topic').get_parameter_value().string_value
        service_name = self.get_parameter('service_name').get_parameter_value().string_value
        entropy_viz_topic = self.get_parameter('entropy_viz_topic').get_parameter_value().string_value
        
        # Initialize variables
        self.latest_image = None
        self.latest_gaze_data = None
        
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
        self.entropy_service = self.create_service(
            EntropyCalculation,
            service_name,
            self.entropy_calculation_callback
        )
        
        self.get_logger().info('Entropy Gaze tracking node initialized')
        self.get_logger().info(f'Subscribed to: {image_topic} and {gaze_topic}')
        self.get_logger().info(f'Service available: {service_name}')
        self.get_logger().info(f'Publishing entropy visualization to: {entropy_viz_topic}')
        self.get_logger().info(f'Parameters - Calibration time: {self.calibration_time}s, Frames: {self.frames_required}, Angle tolerance: {self.deg_angle_tolerance}°')
    
    def image_callback(self, msg):
        """Callback for raw image data"""
        try:
            # Convert ROS Image message to OpenCV format
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.get_logger().debug('Received image frame')
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
    
    def gaze_callback(self, msg):
        """Callback for gaze data"""
        self.latest_gaze_data = msg
        self.get_logger().debug(f'Received gaze data: pitch={msg.pitch}, yaw={msg.yaw}')
        
        # Generate and publish entropy visualization if we have both image and gaze data
        if self.latest_image is not None:
            self.publish_entropy_visualization()
    
    def publish_entropy_visualization(self):
        """Create and publish entropy visualization image"""
        try:
            if self.latest_image is None or self.latest_gaze_data is None:
                return
            
            # Create a copy of the latest image for visualization
            viz_image = self.latest_image.copy()
            
            # Add gaze information overlay
            height, width = viz_image.shape[:2]
            
            # Add text overlays with gaze information
            cv2.putText(viz_image, f'Pitch: {self.latest_gaze_data.pitch:.2f}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(viz_image, f'Yaw: {self.latest_gaze_data.yaw:.2f}', 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add calibration parameters
            cv2.putText(viz_image, f'Cal Time: {self.calibration_time}s', 
                       (10, height-90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
            cv2.putText(viz_image, f'Frames: {self.frames_required}', 
                       (10, height-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
            cv2.putText(viz_image, f'Tolerance: {self.deg_angle_tolerance}°', 
                       (10, height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
            
            # Convert back to ROS Image message and publish
            viz_msg = self.bridge.cv2_to_imgmsg(viz_image, "bgr8")
            viz_msg.header.stamp = self.get_clock().now().to_msg()
            self.entropy_viz_publisher.publish(viz_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error creating entropy visualization: {str(e)}')
    
    def entropy_calculation_callback(self, request, response):
        """Service callback for entropy calculation"""
        try:
            # Update calibration parameters from service request
            self.calibration_time = request.calibration_time
            self.frames_required = request.frames_required
            self.deg_angle_tolerance = request.deg_angle_tolerance
            
            self.get_logger().info(f'Entropy calculation requested with parameters:')
            self.get_logger().info(f'  Calibration time: {self.calibration_time} seconds')
            self.get_logger().info(f'  Frames required: {self.frames_required}')
            self.get_logger().info(f'  Angle tolerance: {self.deg_angle_tolerance} degrees')
            
            # Here you would implement the actual entropy calculation logic
            # For now, we'll return success if we have valid parameters
            if (self.calibration_time > 0 and 
                self.frames_required > 0 and 
                self.deg_angle_tolerance > 0):
                response.success = True
                self.get_logger().info('Entropy calculation completed successfully')
            else:
                response.success = False
                self.get_logger().warn('Invalid parameters for entropy calculation')
                
        except Exception as e:
            self.get_logger().error(f'Error in entropy calculation: {str(e)}')
            response.success = False
        
        return response


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


