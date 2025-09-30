#!/bin/python3

import math
import cv2
import numpy as np
from time import time, sleep
from collections import deque
from threading import Thread, Lock

class AttentionDetector:
    def __init__(self, 
                 attention_threshold=0.5,  # Time in seconds needed to confirm attention
                 pitch_threshold=15,       # Maximum pitch angle for attention
                 yaw_threshold=20,         # Maximum yaw angle for attention
                 history_size=10):         # Number of frames to keep for smoothing

        # Initialize parameters
        self.attention_threshold = attention_threshold
        self.pitch_threshold = pitch_threshold
        self.yaw_threshold = yaw_threshold
        self.history_size = history_size
        
        # Track per-ID data
        self.id_data = {}  # Dictionary to store per-ID attention tracking data
        
        # Define colors for different IDs
        self.colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
        ]
    
    def _initialize_id_data(self, id_num):
        """Initialize tracking data for a new ID"""
        if id_num not in self.id_data:
            self.id_data[id_num] = {
                'angle_history': deque(maxlen=self.history_size),
                'attention_start_time': None,
                'attention_state': False
            }
    
    def smooth_angles(self, angles, id_num):
        """Apply smoothing to angles using a moving average for specific ID"""
        self._initialize_id_data(id_num)
        self.id_data[id_num]['angle_history'].append(angles)
        return np.mean(self.id_data[id_num]['angle_history'], axis=0)
    
    def is_looking_at_robot(self, pitch, yaw):
        """Determine if the person is looking at the robot based on angles"""
        return abs(pitch) < self.pitch_threshold and abs(yaw) < self.yaw_threshold
    
    def process_frame(self, frame, gaze_results):
        """
        Process a single frame and return attention state and visualization
        
        Args:
            frame: RGB image frame
            gaze_results: GazeFrame object containing array of GazeDetection objects
            
        Returns:
            frame: Visualized frame with attention overlays
            id_results: Dictionary with ID as key and attention data as values
        """
        # Initialize return dictionary
        id_results = {}
        
        # Process each gaze detection in the results
        if gaze_results and hasattr(gaze_results, 'gazes'):
            for gaze_detection in gaze_results.gazes:
                id_num = gaze_detection.id
                
                # Initialize tracking for this ID if needed
                self._initialize_id_data(id_num)
                
                # Extract angles (convert from radians to degrees if needed)
                pitch = gaze_detection.pitch
                yaw = gaze_detection.yaw
                
                # Create angles tuple (pitch, yaw, roll=0)
                angles = (pitch, yaw, 0)
                
                # Apply smoothing for this ID
                smoothed_angles = self.smooth_angles(angles, id_num)
                smoothed_pitch, smoothed_yaw, _ = smoothed_angles
                
                # Check if looking at robot
                attention_detected = self.is_looking_at_robot(smoothed_pitch, smoothed_yaw)
                
                # Track sustained attention for this ID
                current_time = time()
                sustained_attention = False
                
                if attention_detected:
                    if self.id_data[id_num]['attention_start_time'] is None:
                        self.id_data[id_num]['attention_start_time'] = current_time
                    elif (current_time - self.id_data[id_num]['attention_start_time']) >= self.attention_threshold:
                        sustained_attention = True
                else:
                    self.id_data[id_num]['attention_start_time'] = None
                
                # Store results for this ID
                id_results[id_num] = {
                    'attention_detected': attention_detected,
                    'sustained_attention': sustained_attention,
                    'angles': smoothed_angles,
                    'bounding_box': {
                        'x': gaze_detection.bounding_box_x,
                        'y': gaze_detection.bounding_box_y,
                        'width': gaze_detection.bounding_box_width,
                        'height': gaze_detection.bounding_box_height
                    },
                    'raw_pitch': pitch,
                    'raw_yaw': yaw
                }
                
                # Draw visualization for this detection
                frame = self._draw_attention_visualization(frame, gaze_detection, id_results[id_num])
        
        return frame, id_results
    
    def _draw_attention_visualization(self, frame, gaze_detection, result_data):
        """Draw attention visualization for a single detection"""
        id_num = gaze_detection.id
        
        # Get bounding box coordinates
        x = int(gaze_detection.bounding_box_x)
        y = int(gaze_detection.bounding_box_y)
        width = int(gaze_detection.bounding_box_width)
        height = int(gaze_detection.bounding_box_height)
        
        x_min, y_min = x, y
        x_max, y_max = x + width, y + height
        
        # Use different color for each ID
        base_color = self.colors[(id_num - 1) % len(self.colors)]
        
        # Determine attention color overlay
        if result_data['sustained_attention']:
            attention_color = (0, 255, 0)  # Green for sustained attention
            text_bg_color = (0, 200, 0)
            status_text = "SUSTAINED"
        elif result_data['attention_detected']:
            attention_color = (0, 255, 255)  # Yellow for attention detected
            text_bg_color = (0, 200, 200)
            status_text = "ATTENTION"
        else:
            attention_color = (0, 0, 255)  # Red for no attention
            text_bg_color = (0, 0, 200)
            status_text = "NO ATTENTION"
        
        # Draw bounding box with ID-specific color
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), base_color, 3)
        
        # Draw attention status overlay on bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), attention_color, 2)
        
        # Draw gaze direction arrow
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        
        # Calculate arrow endpoint based on pitch and yaw
        arrow_length = 60
        pitch = result_data['raw_pitch']
        yaw = result_data['raw_yaw']
        
        # Calculate arrow direction
        dx = -arrow_length * math.sin(pitch) * math.cos(yaw)
        dy = -arrow_length * math.sin(yaw)
        
        end_x = int(center_x + dx)
        end_y = int(center_y + dy)
        
        cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), base_color, 4)
        
        # Add text labels with ID-specific background
        pitch_deg = math.degrees(pitch)
        yaw_deg = math.degrees(yaw)
        label_text = f"ID:{id_num} P:{pitch_deg:.1f} Y:{yaw_deg:.1f}"
        status_label = f"{status_text}"
        
        # Draw background rectangle for main text
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x_min, y_min - text_height - 10), 
                     (x_min + text_width, y_min), base_color, -1)
        
        # Draw main text
        cv2.putText(frame, label_text, (x_min, y_min - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw attention status text
        (status_width, status_height), _ = cv2.getTextSize(status_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x_min, y_max), 
                     (x_min + status_width, y_max + status_height + 5), text_bg_color, -1)
        
        cv2.putText(frame, status_label, (x_min, y_max + status_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame


