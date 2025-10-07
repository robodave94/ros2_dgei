std_msgs/Header header
# Add gaze detection message
GazeDetection gaze_detection

# Gaze analysis fields
bool attention_state
bool sustained_attention
float64 smoothed_pitch
float64 smoothed_yaw
float64 original_pitch
float64 original_yaw
float64 gaze_score
int robot_looks
float64 gaze_entropy