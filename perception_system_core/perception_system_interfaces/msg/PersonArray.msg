# The people of one image, published by the body detector on /people/body.

# The stamp and the camera frame of the image
std_msgs/Header header
# The frame of the 3D positions (odom by default); empty if nothing is placed in 3D
string fixed_frame_id
Person[] people
