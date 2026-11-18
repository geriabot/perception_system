# A keypoint of a person's body (COCO order, in Person.keypoints): nose, left_eye, right_eye,
# left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist,
# right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle. Left and right
# are the person's.

string name
# In pixels of the image
float32 x
float32 y
# Confidence of the model, in [0, 1]; below the body detector's keypoint_threshold it is not used
float32 score
# Its position in the fixed frame (PersonArray.fixed_frame_id) at the time of the image; false
# when the keypoint is not used or the system cannot place it in 3D
bool has_position
geometry_msgs/Point position
