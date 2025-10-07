std_msgs/Header header
uint8 id
# Add gaze detection message
GazeDetection gaze_detection

# Gaze analysis fields
bool attention_state
bool sustained_attention
float64 smoothed_pitch
float64 smoothed_yaw
float64 gaze_score
int32 robot_looks
float64 gaze_entropy