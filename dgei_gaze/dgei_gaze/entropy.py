#!/usr/bin/python3.10

from l2cs.gaze_detectors import Gaze_Detector
import torch
from copy import deepcopy
import math
import pdb
import cv2
import numpy as np
from time import time, sleep
from collections import deque
from threading import Thread, Lock


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



# Everything in this file is in degrees, please ensure inputs are converted if necessary

class AttentionTracker:
    def __init__(self,
                 face_id,
                 attention_threshold=0.5,  # Time in seconds needed to confirm attention
                 history_size=10,         # Number of frames to keep for smoothing)
                 # Calibration baselines (assumed to be looking at robot)
                 cal_dict = {}):
        # Initialize parameters
        self.face_id = face_id
        self.attention_threshold = attention_threshold
        # self.pitch_threshold = pitch_threshold
        # self.yaw_threshold = yaw_threshold

        self.update_calibration_parameters(cal_dict)

        self.attention_start_time = None
        self.attention_state = False

        # Initialize angle history for smoothing
        self.angle_history = deque(maxlen=history_size)

        self.attention_window = []  # List of (timestamp, attention_state) tuples
        
    def append_to_attention_window(self, current_time, attention):
        self.attention_window.append((current_time, attention))

        # Limit the size of the attention window to the last 1000 entries
        if len(self.attention_window) > 1000:
            self.attention_window = self.attention_window[-1000:]


    def get_attention_window(self):
        return self.attention_window

    def update_calibration_parameters(self, cal_dict):
        # Calibration parameters
        if 'BaselinePitch' in cal_dict:
            self.cal_baseline_pitch = cal_dict['BaselinePitch']
        else:
            raise ValueError("Calibration dictionary must contain 'BaselinePitch'")

        if 'BaselineYaw' in cal_dict:
            self.cal_baseline_yaw = cal_dict['BaselineYaw']
        else:
            raise ValueError("Calibration dictionary must contain 'BaselineYaw'")

        if 'PitchThreshold' in cal_dict:
            self.cal_pitch_threshold = cal_dict['PitchThreshold']
        else:
            raise ValueError("Calibration dictionary must contain 'PitchThreshold'")

        if 'YawThreshold' in cal_dict:
            self.cal_yaw_threshold = cal_dict['YawThreshold']
        else:
            raise ValueError("Calibration dictionary must contain 'YawThreshold'")

        return True

    def get_track_id(self):
        """Return the face ID being tracked"""
        return self.face_id

    def smooth_angles(self, angles):
        """Apply smoothing to angles using a moving average"""
        self.angle_history.append(angles)
        return np.mean(self.angle_history, axis=0)
    
    def is_looking_at_robot(self, pitch, yaw):
        """Determine if the person is looking at the robot based on angles"""
        # Calculate angle differences from baseline
        pitch_diff = abs(pitch - self.cal_baseline_pitch)
        yaw_diff = abs(yaw - self.cal_baseline_yaw)

        return pitch_diff < self.cal_pitch_threshold and yaw_diff < self.cal_yaw_threshold

    def process_angle_frame_update(self, pitch, yaw):
        """Process a new frame of pitch and yaw angles"""
        smoothed_pitch, smoothed_yaw = self.smooth_angles((pitch, yaw))

        current_time = time()
        attention = self.is_looking_at_robot(smoothed_pitch, smoothed_yaw)
        if attention:
            if self.attention_start_time is None:
                self.attention_start_time = current_time
            elif (current_time - self.attention_start_time) >= self.attention_threshold:
                self.attention_state = True # Attention state is sustained_attention
        else:
            self.attention_start_time = None
            self.attention_state = False # Attention state is sustained_attention
        
        # Returns attention, sustained attention state, smoothed pitch, smoothed yaw
        return attention, self.attention_state, smoothed_pitch, smoothed_yaw

class AttentionTrackerCollection:
    """Collection class to manage multiple AttentionTracker objects"""
    
    def __init__(self, 
                 calibration_dictionary,
                 default_attention_threshold=0.5,
                 default_history_size=10,
                 auto_cleanup=True,
                 cleanup_timeout=3.0):  # Remove trackers inactive for 3 seconds

        self.trackers = {}  # Dictionary to store trackers by face_id
        
        # Default parameters for new trackers
        self.default_attention_threshold = default_attention_threshold
        self.calibration_dictionary = calibration_dictionary
        self.default_history_size = default_history_size
        
        # Auto cleanup settings
        self.auto_cleanup = auto_cleanup
        self.cleanup_timeout = cleanup_timeout
        self.last_update_times = {}  # Track when each tracker was last used
        
        # Thread safety
        self.lock = Lock()
    
    def update_calibration_dictionary(self, cal_dict):
        """Update the calibration dictionary for all trackers"""
        with self.lock:
            self.calibration_dictionary = cal_dict
            for tracker in self.trackers.values():
                tracker.update_calibration_parameters(cal_dict)

    def get_tracker(self, face_id):
        """Get tracker by face ID, create if doesn't exist"""
        with self.lock:
            if face_id not in self.trackers:
                self.trackers[face_id] = AttentionTracker(
                    face_id=face_id,
                    attention_threshold=self.default_attention_threshold,
                    history_size=self.default_history_size,
                    cal_dict=self.calibration_dictionary
                )
            
            # Update last access time
            self.last_update_times[face_id] = time()
            return self.trackers[face_id]
    
    def has_tracker(self, face_id):
        """Check if tracker exists for given face ID"""
        with self.lock:
            return face_id in self.trackers
    
    def remove_tracker(self, face_id):
        """Remove tracker by face ID"""
        with self.lock:
            if face_id in self.trackers:
                del self.trackers[face_id]
                if face_id in self.last_update_times:
                    del self.last_update_times[face_id]
                return True
            return False
    
    def get_all_trackers(self):
        """Get all trackers as a dictionary {face_id: tracker}"""
        with self.lock:
            return dict(self.trackers)  # Return copy to avoid external modification
    
    def get_all_tracker_ids(self):
        """Get list of all face IDs being tracked"""
        with self.lock:
            return list(self.trackers.keys())
    
    def get_tracker_count(self):
        """Get number of active trackers"""
        with self.lock:
            return len(self.trackers)
    
    def process_frame_data(self, face_detections):
        """
        Process frame data for multiple faces
        
        Args:
            face_detections: List of dicts with keys 'id', 'pitch', 'yaw'
                           e.g. [{'id': 1, 'pitch': 10.5, 'yaw': -5.2}, ...]
        
        Returns:
            dict: Results for each face {face_id: (attention_state, pitch, yaw)}
        """
        results = {}
        
        for detection in face_detections:
            face_id = detection['id']
            pitch = detection['pitch']
            yaw = detection['yaw']
            
            # Get or create tracker
            tracker = self.get_tracker(face_id)
            
            # Process the frame
            attention, sustained_attention, smoothed_pitch, smoothed_yaw = tracker.process_angle_frame_update(pitch, yaw)
            
            # Update attention window
            current_time = time()
            tracker.append_to_attention_window(current_time, attention)

            # Calculate metrics
            metrics = calculate_attention_metrics(tracker.get_attention_window())

            results[face_id] = {
                'attention_state': attention,
                'sustained_attention': sustained_attention,
                'smoothed_pitch': smoothed_pitch,
                'smoothed_yaw': smoothed_yaw,
                'original_pitch': pitch,
                'original_yaw': yaw,
                'gaze_score': metrics['gaze_score'],
                'robot_looks': metrics['robot_looks'],
                'gaze_entropy': metrics['gaze_entropy']
            }
        
        # Perform cleanup if enabled
        if self.auto_cleanup:
            self._cleanup_inactive_trackers()
        
        return results
    
    def get_attention_states(self):
        """Get current attention states for all trackers"""
        with self.lock:
            return {face_id: tracker.attention_state 
                   for face_id, tracker in self.trackers.items()}
    
    def get_trackers_with_attention(self):
        """Get list of face IDs that currently have attention"""
        attention_states = self.get_attention_states()
        return [face_id for face_id, has_attention in attention_states.items() if has_attention]
    
    def clear_all_trackers(self):
        """Remove all trackers"""
        with self.lock:
            self.trackers.clear()
            self.last_update_times.clear()
    
    def _cleanup_inactive_trackers(self):
        """Remove trackers that haven't been updated recently (internal method)"""
        current_time = time()
        inactive_ids = []
        
        with self.lock:
            for face_id, last_update in self.last_update_times.items():
                if current_time - last_update > self.cleanup_timeout:
                    inactive_ids.append(face_id)
            
            # Remove inactive trackers
            for face_id in inactive_ids:
                if face_id in self.trackers:
                    del self.trackers[face_id]
                del self.last_update_times[face_id]
    
    def manual_cleanup(self, timeout=None):
        """Manually trigger cleanup of inactive trackers"""
        if timeout is None:
            timeout = self.cleanup_timeout
            
        current_time = time()
        inactive_ids = []
        
        with self.lock:
            for face_id, last_update in self.last_update_times.items():
                if current_time - last_update > timeout:
                    inactive_ids.append(face_id)
            
            # Remove inactive trackers
            for face_id in inactive_ids:
                if face_id in self.trackers:
                    del self.trackers[face_id]
                del self.last_update_times[face_id]
        
        return inactive_ids  # Return list of removed face IDs
    
    def set_default_parameters(self, **kwargs):
        """Update default parameters for new trackers"""
        if 'attention_threshold' in kwargs:
            self.default_attention_threshold = kwargs['attention_threshold']
        if 'pitch_threshold' in kwargs:
            self.default_pitch_threshold = kwargs['pitch_threshold']
        if 'yaw_threshold' in kwargs:
            self.default_yaw_threshold = kwargs['yaw_threshold']
        if 'history_size' in kwargs:
            self.default_history_size = kwargs['history_size']
    
    def update_tracker_parameters(self, face_id, **kwargs):
        """Update parameters for a specific tracker"""
        with self.lock:
            if face_id in self.trackers:
                tracker = self.trackers[face_id]
                if 'attention_threshold' in kwargs:
                    tracker.attention_threshold = kwargs['attention_threshold']
                if 'pitch_threshold' in kwargs:
                    tracker.pitch_threshold = kwargs['pitch_threshold']
                if 'yaw_threshold' in kwargs:
                    tracker.yaw_threshold = kwargs['yaw_threshold']
                return True
            return False
    
    def __len__(self):
        """Return number of trackers"""
        return self.get_tracker_count()
    
    def __contains__(self, face_id):
        """Check if face_id exists in collection"""
        return self.has_tracker(face_id)
    
    def __getitem__(self, face_id):
        """Get tracker by face_id using [] operator"""
        return self.get_tracker(face_id)
    
    def __iter__(self):
        """Iterate over face_ids"""
        with self.lock:
            return iter(list(self.trackers.keys()))
        

# Calibration Class HERE
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

    def get_calibrated_parameters(self):
        """Return calibrated parameters if calibration is complete"""
        if not self.is_calibrated:
            raise ValueError("Calibration not yet complete")
        
        return {
            'BaselinePitch': self.baseline_pitch,
            'BaselineYaw': self.baseline_yaw,
            'PitchThreshold': self.pitch_threshold,
            'YawThreshold': self.yaw_threshold
        }
        
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