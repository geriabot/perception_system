# An object that a detection request sees in an image

# The tracker id (empty while the track is not confirmed, or without a tracker)
string id
# The class, as the request publishes it
string class_name
float32 score
# The box, in pixels of the image
vision_msgs/BoundingBox2D bbox
# Its position in the fixed frame (fixed_frame_id, odom by default) at the time of the image;
# has_position is false when the system places no detection in 3D or the transform is missing
bool has_position
geometry_msgs/PointStamped position
