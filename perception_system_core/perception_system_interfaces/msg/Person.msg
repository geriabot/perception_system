# A person seen by the body detector (the skill "body"), in one image.

# The tracker id (each person keeps it while they are tracked)
string id
float32 score
# The box, in pixels of the image
vision_msgs/BoundingBox2D bbox
# The 17 keypoints of the body, COCO order (BodyKeypoint)
BodyKeypoint[] keypoints

# standing, sitting, lying or unknown (too few keypoints)
string posture
# Lying on the floor after a fall, for longer than the detector's fall_confirm_s (with 3D: the
# torso below fall_max_height_m; a person lying on a bed is not fallen)
bool fallen
# The gestures being made, held for gesture_hold_s: hand_raised_left, hand_raised_right,
# both_hands_raised, waving, pointing_left, pointing_right (the person's left and right)
string[] gestures
# The face is turned to the camera (eyes, nose and ears seen symmetrically)
bool facing_camera
# The dominant color of the upper body: red, orange, yellow, green, blue, purple, pink, brown,
# black, white, gray, or empty if the torso is not seen
string clothing_color

# The middle of the torso in the fixed frame
bool has_position
geometry_msgs/Point position
# The head (nose, or the middle of the eyes or ears) in the fixed frame
bool has_head
geometry_msgs/Point head
# While pointing: the direction of the arm (shoulder to wrist) in the fixed frame, unit length,
# and where it meets the floor (z = the detector's floor_z), if it does within max_pointing_m
bool pointing
geometry_msgs/Vector3 pointing_direction
bool has_pointed_location
geometry_msgs/Point pointed_location
# The TF frames published for this person, children of the fixed frame (empty if none):
# <prefix>_<id>, <prefix>_<id>_head, <prefix>_<id>_pointed
string frame_id
