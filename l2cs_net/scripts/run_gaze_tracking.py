#!/bin/python3

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter


class GazeTrackingNode(Node):
    def __init__(self):
        super().__init__('gaze_tracking_node')
        
        # Declare parameters with default values
        self.declare_parameter('camera_number', 0)  # Default to first camera
        self.declare_parameter('visualisation_topic', '/gaze/frame') 
        self.declare_parameter('model_path', '/path/to/model.pth')
        self.declare_parameter('device', 'cuda')
        
        # Get parameter values
        self.camera_number = self.get_parameter('camera_number').get_parameter_value().integer_value
        self.visualisation_topic = self.get_parameter('visualisation_topic').get_parameter_value().string_value
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        
        # Log the parameters
        self.get_logger().info(f'Camera topic: {self.camera_topic}')
        self.get_logger().info(f'Output topic: {self.output_topic}')
        self.get_logger().info(f'Model path: {self.model_path}')
        self.get_logger().info(f'Device: {self.device}')
        self.get_logger().info(f'Confidence threshold: {self.confidence_threshold}')
        self.get_logger().info(f'Publish rate: {self.publish_rate}')
        
        # Initialize your gaze tracking components here
        self.setup_gaze_tracking()
        
        # Create timer for main processing loop
        self.timer = self.create_timer(1.0 / self.publish_rate, self.process_frame)
        
    def setup_gaze_tracking(self):
        """Initialize gaze tracking model and subscribers/publishers"""
        # TODO: Initialize your L2CS model here
        # TODO: Create image subscriber
        # TODO: Create gaze direction publisher
        pass
        
    def process_frame(self):
        """Main processing loop"""
        # TODO: Process camera frame and publish gaze direction
        pass


def main(args=None):
    rclpy.init(args=args)
    
    try:
        gaze_node = GazeTrackingNode()
        rclpy.spin(gaze_node)
    except KeyboardInterrupt:
        pass
    finally:
        gaze_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()