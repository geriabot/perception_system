# The state of the perception system, published on /perception_system/status on every change
# and once per second (reliable, transient local)

builtin_interfaces/Time stamp
# Lifecycle state of /perception_system: unconfigured, inactive, active, finalized
string state
RequestStatus[] requests
# Detection, over the last stats period: images processed per second, median milliseconds per
# image (all the open detection requests, the 3D placement included), failed images
float32 images_per_second
float32 ms_per_image
uint32 failures
# The detection models loaded (local) or the server sessions open (remote)
string[] models
# The skills whose node is active
string[] active_skills
