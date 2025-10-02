
from copy import deepcopy
import math
import pdb
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

class AttentionCalibrator:
    def __init__(self, 
                 calibration_time=10.0,    # Time in seconds needed for calibration
                 samples_needed=300,        # Number of samples to collect
                 angle_tolerance=15.0):     # Tolerance for angle variation during calibration -- DEFAULT 15.0
        
        self.calibration_time = calibration_time
        self.samples_needed = samples_needed
        self.angle_tolerance = angle_tolerance
        
        # Storage for calibration samples
        self.pitch_samples = []
        self.yaw_samples = []
        
        # Calibration state
        self.calibration_start_time = None
        self.is_calibrated = False
        self.baseline_pitch = None
        self.baseline_yaw = None
        self.pitch_threshold = None
        self.yaw_threshold = None
        
    def start_calibration(self):
        """Start the calibration process"""
        self.calibration_start_time = time()
        self.pitch_samples = []
        self.yaw_samples = []
        self.is_calibrated = False
        print("Starting calibration... Please look directly at the robot.")
        
    def process_calibration_frame(self, pitch, yaw):
        """Process a frame during calibration"""
        if self.calibration_start_time is None:
            return False, "Calibration not started"
        
        current_time = time()
        elapsed_time = current_time - self.calibration_start_time
        
        # Add samples
        self.pitch_samples.append(pitch)
        self.yaw_samples.append(yaw)
        
        # Check if we have enough samples
        if len(self.pitch_samples) >= self.samples_needed:
            # Calculate baseline angles and thresholds
            self.baseline_pitch = np.mean(self.pitch_samples)
            self.baseline_yaw = np.mean(self.yaw_samples)
            
            # Calculate standard deviations
            pitch_std = np.std(self.pitch_samples)
            yaw_std = np.std(self.yaw_samples)
            
            # Set thresholds based on standard deviation and minimum tolerance
            self.pitch_threshold = max(2 * pitch_std, self.angle_tolerance)
            self.yaw_threshold = max(2 * yaw_std, self.angle_tolerance)
            
            self.is_calibrated = True
            return True, "Calibration complete"
        
        # Still calibrating
        return False, f"Calibrating... {len(self.pitch_samples)}/{self.samples_needed} samples"

class CalibratedAttentionDetector(AttentionDetector):
    def __init__(self, calibrator, attention_threshold=0.5, history_size=10):
        super().__init__(
            attention_threshold=attention_threshold,
            pitch_threshold=None,  # Will be set by calibrator
            yaw_threshold=None,    # Will be set by calibrator
            history_size=history_size
        )
        
        # Store calibrator
        self.calibrator = calibrator
        
        # Set thresholds from calibrator
        if calibrator.is_calibrated:
            self.pitch_threshold = calibrator.pitch_threshold
            self.yaw_threshold = calibrator.yaw_threshold
            self.baseline_pitch = calibrator.baseline_pitch
            self.baseline_yaw = calibrator.baseline_yaw
    
    def is_looking_at_robot(self, pitch, yaw):
        """Override the original method to use calibrated values"""
        if not self.calibrator.is_calibrated:
            return False
            
        # Calculate angle differences from baseline
        pitch_diff = abs(pitch - self.calibrator.baseline_pitch)
        yaw_diff = abs(yaw - self.calibrator.baseline_yaw)
        
        return pitch_diff < self.calibrator.pitch_threshold and yaw_diff < self.calibrator.yaw_threshold

def calculate_attention_metrics(attention_window, interval_duration=5.0):
    """
    Calculate attention metrics for a given time window of attention data.
    
    Args:
        attention_window (list): List of tuples (timestamp, attention_state)
        interval_duration (float): Duration of analysis interval in seconds
        
    Returns:
        dict: Dictionary containing attention metrics:
            - attention_ratio: Ratio of frames with attention detected
            - gaze_entropy: Shannon entropy of gaze distribution
            - frames_in_interval: Number of frames in analyzed interval
            - robot_looks: Number of frames looking at robot
            - non_robot_looks: Number of frames not looking at robot
    """
    if not attention_window:
        return {
            'attention_ratio': 0.0,
            'gaze_entropy': 0.0,
            'frames_in_interval': 0,
            'robot_looks': 0,
            'non_robot_looks': 0,
            'gaze_score': 0.0
        }
    
    # Get current time and filter window to only include last interval_duration seconds
    current_time = attention_window[-1][0]  # Latest timestamp
    filtered_window = [(t, a) for t, a in attention_window 
                      if current_time - t <= interval_duration]
    # print("The filtered window is ", filtered_window)
    # print("the size of the filtered window is ", len(filtered_window))
    
    # Count frames
    frames_in_interval = len(filtered_window)
    robot_looks = sum(1 for _, attention in filtered_window if attention)
    non_robot_looks = frames_in_interval - robot_looks
    
    # Calculate attention ratio
    attention_ratio = robot_looks / frames_in_interval if frames_in_interval > 0 else 0.0
    
    # Calculate stationary gaze entropy
    gaze_entropy = 0.0
    if frames_in_interval > 0:
        p_robot = robot_looks / frames_in_interval
        p_non_robot = non_robot_looks / frames_in_interval
        
        # Calculate entropy using Shannon formula
        if p_robot > 0:
            gaze_entropy -= p_robot * math.log2(p_robot)
        if p_non_robot > 0:
            gaze_entropy -= p_non_robot * math.log2(p_non_robot)
    
    ### Cacluations for gaze score on pepper

    # Compute gaze score using the new formula
    # # High gc Low entrophy: gaze score high 
    # # High gc High entropy: gaze score low
    # # low gc high entrophy: gaze score low
    # # low gc low entrophy: gaze score low
    
    # Normalize the attention ratio
    normalized_attention_ratio = min(attention_ratio, 1.0)
    
    # Normalize the gaze entropy (lower entropy is better for focused attention)
    normalized_entropy = 1.0 - min(gaze_entropy, 1.0)

    if gaze_entropy == 1.0 or (robot_looks > 30 and 1.0 > gaze_entropy > 0.7):
        gaze_score = 100 * normalized_attention_ratio
    else:
        gaze_score = 100 * (normalized_attention_ratio * normalized_entropy)
    
    gaze_score = max(0, min(100, gaze_score))  # Ensure score is within 0-100
    
    ### End additional calculations
    
    return {
        'attention_ratio': attention_ratio,
        'gaze_entropy': gaze_entropy,
        'frames_in_interval': frames_in_interval,
        'robot_looks': robot_looks,
        'non_robot_looks': non_robot_looks,
        'gaze_score': gaze_score
    }


def attention_detection_loop(self):
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_id)
        
        # Initialize camera and detector with calibration
        self.detector = CalibratedAttentionDetector(self.calibrator)
        
        self.is_in_attention_detection_mode = True
        
        
        self.attention_window = []
        
        while self.cap.isOpened() and self.is_in_attention_detection_mode:
            success, frame = self.cap.read()
            if not success:
                print("Failed to read frame. Stopping attention detection.")
                break
                
            # print("Processing frame")
            # Process frame
            frame, attention, sustained, angles, face_found = self.detector.process_frame(frame)
            
            # Update attention window
            current_time = time()
            self.attention_window.append((current_time, attention))
            
            # Calculate metrics
            metrics = calculate_attention_metrics(self.attention_window)
            
            self.gaze_score_lock.acquire()
            self.gaze_score = metrics["gaze_score"]
            self.gaze_score_lock.release()
            
            self.robot_looks_lock.acquire()
            self.robot_looks = metrics["robot_looks"]
            self.robot_looks_lock.release()
            
            self.gaze_entropy_lock.acquire()
            self.gaze_entropy = metrics["gaze_entropy"]
            self.gaze_entropy_lock.release()
            
            # Add metrics and calibration values to display
            if face_found:
                h, w, _ = frame.shape
                # Add calibration values
                cv2.putText(frame, f'Baseline Pitch: {self.calibrator.baseline_pitch:.1f}', 
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                cv2.putText(frame, f'Baseline Yaw: {self.calibrator.baseline_yaw:.1f}', 
                        (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                
                # Add metrics
                cv2.putText(frame, f'Attention Ratio: {metrics["attention_ratio"]:.2f}', 
                        (20, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                cv2.putText(frame, f'Gaze Entropy: {metrics["gaze_entropy"]:.2f}', 
                        (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                cv2.putText(frame, f'Frames in Window: {metrics["frames_in_interval"]}', 
                        (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                
                self.gaze_score_lock.acquire()
                self.visualisation_frame = frame
                self.gaze_score_lock.release()