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
        self.attention_start_time = None
        self.attention_state = False
        
        # Initialize angle history for smoothing
        self.angle_history = deque(maxlen=history_size)
        
    
    def smooth_angles(self, angles):
        """Apply smoothing to angles using a moving average"""
        self.angle_history.append(angles)
        return np.mean(self.angle_history, axis=0)
    
    def is_looking_at_robot(self, pitch, yaw):
        """Determine if the person is looking at the robot based on angles"""
        return abs(pitch) < self.pitch_threshold and abs(yaw) < self.yaw_threshold
    
    def process_frame(self, frame):
        # Initialize return values
        attention_detected = False
        sustained_attention = False
        angles = None
        face_found = False

        """Process a single frame and return attention state and visualization"""
        h, w, _ = frame.shape
        
        g_success = self.gaze_detector.detect_gaze(frame)

        if g_success:
            face_found = True
            results = self.gaze_detector.get_latest_gaze_results()
            if results is None:
                return frame, attention_detected, sustained_attention, angles, face_found

            # Extract pitch and yaw from results
            pitch = results.pitch[0]
            yaw = results.yaw[0]

            angles = (math.degrees(pitch), math.degrees(yaw), 0) # Assuming roll is not used and angles are in degrees
            
            # Apply smoothing
            smoothed_angles = self.smooth_angles(angles)
            pitch, yaw, _ = smoothed_angles
            
            # Check if looking at robot
            attention_detected = self.is_looking_at_robot(pitch, yaw)
            
            # Track sustained attention
            current_time = time()
            if attention_detected:
                if self.attention_start_time is None:
                    self.attention_start_time = current_time
                elif (current_time - self.attention_start_time) >= self.attention_threshold:
                    sustained_attention = True
            else:
                self.attention_start_time = None

            frame = self.gaze_detector.draw_gaze_window()

            # Visualization
            color = (0, 255, 0) if sustained_attention else (
                (0, 165, 255) if attention_detected else (0, 0, 255)
            )
            
            # Add text overlays
            cv2.putText(frame, f'Pitch: {int(pitch)}', (20, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f'Yaw: {int(yaw)}', (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw attention status
            status = "Sustained Attention" if sustained_attention else (
                "Attention Detected" if attention_detected else "No Attention"
            )
            cv2.putText(frame, status, (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            
                
        return frame, attention_detected, sustained_attention, angles, face_found
