# Something a person started doing, published by the body detector on /people/events.

# The gesture that started (hand_raised_left, hand_raised_right, both_hands_raised, waving,
# pointing_left, pointing_right), a new posture (standing, sitting, lying), fall (the person is
# fallen) or recovered (no longer fallen)
string event
std_msgs/Header header
string fixed_frame_id
# The person, as in that image
Person person
