# The perception of the body

The skill `body` is the lifecycle node `/body_detector` of `perception_body`, started by the
launch file (`body:=true`, the default) and activated by `/perception_system` while a client
requests it (`start_skill`, the tasks `WaitForGesture` and `WatchForFall`) or from activation
(`autostart_skills`). While inactive it processes no image and loads no model.

For every image it finds the people with the 17 keypoints of their body (YOLO pose, Ultralytics),
gives each one a tracker id (ByteTrack), and reads from the keypoints:

| What | Values | How |
|---|---|---|
| Posture | `standing`, `sitting`, `lying`, `unknown` | The torso more than `lying_torso_angle_deg` (55°) from the vertical is lying. Otherwise, with the knees in view, a thigh more than `sitting_thigh_angle_deg` (50°) from the vertical, or shorter than `sitting_thigh_ratio` (0.5) of the torso (a thigh towards the camera), is sitting. Without the knees, the height of the hips over the floor in 3D (below `sitting_max_hip_height_m`, 0.7 m: sitting). Reported after `posture_hold_s` (0.5 s) |
| Fall | `fallen` | Lying for `fall_confirm_s` (1 s), with the torso below `fall_max_height_m` (0.6 m) over the floor when it is placed in 3D (a person lying on a bed is not fallen). It ends (`recovered`) after `recover_s` (1 s) not lying |
| Gestures | `hand_raised_left`, `hand_raised_right`, `both_hands_raised`, `waving`, `pointing_left`, `pointing_right` | A hand is raised with the wrist `hand_raised_margin` (0.3) torso lengths over the shoulder. Waving: a hand over its elbow moving sideways at least `waving_amplitude` (0.12 torso lengths), changing direction `waving_reversals` (2) times within `waving_window_s` (1.5 s). Pointing: the elbow straight (over `pointing_elbow_angle_deg`, 150°), the arm `pointing_min_elevation_deg` (40°) or more from hanging down, and not raised. A gesture is reported after `gesture_hold_s` (0.4 s) and kept `gesture_release_s` (0.5 s); the gestures of a person lying are not read |
| Where they point | `pointed_location` | The line from the shoulder through the wrist, in the fixed frame, where it meets the floor (`floor_z`), up to `max_pointing_m` (8 m) from the hand |
| Facing the camera | `facing_camera` | Both eyes and the nose seen, the nose near the middle of the eyes (within `facing_asymmetry`, 0.3, of half their distance) |
| Clothes | `clothing_color` | The dominant color (red, orange, yellow, green, blue, purple, pink, brown, black, white, gray) of the center of the torso |

Left and right are the person's. Every threshold is a parameter `body.<name>`
([CONFIGURATION.md](CONFIGURATION.md#body_detector)), so that a robot can be tuned without
training anything; the rules are in `perception_body/body.py`, without ROS, and tested on
hand-made skeletons and on photographs.

## Outputs

| Output | Type | Content |
|---|---|---|
| `/people/body` | `PersonArray` | Every person of the image: id, box, keypoints (pixels and, when placed, the fixed frame), posture, `fallen`, gestures, `facing_camera`, `clothing_color`, the torso and the head in the fixed frame, the pointing direction and the place pointed at, and the TF frame |
| `/people/events` | `BodyEvent` | What a person started: a gesture, a posture, `fall`, `recovered`; with the person |
| `/people/body_image` | `sensor_msgs/Image` | The skeletons, ids, postures and gestures (only while someone subscribes) |
| `/tf` | | `body_<id>` (the torso), `body_<id>_head`, and while pointing at the floor `body_<id>_pointed`, children of `fixed_frame_id` (odom), with the stamp of the image |
| `/body_detector/get_people` | `GetPeople` (service) | The people of the last image |

The prefix `body` of the frames keeps them apart from those of the detection requests
(`person_<id>` of `ObserveClass`): the body detector has its own tracker and ids.

## 3D placement

The keypoints are placed in the camera frame and then in the fixed frame with the camera pose at
the time of the image (TF). With a depth image aligned with the color image (`camera:=rgbd`, the
`depth_image` estimator), each keypoint takes the median depth around it; a keypoint far from the
torso (the background beside an arm) takes the torso's. Without one, the whole body is at the
distance that the size of its torso (0.5 m) and shoulders (0.36 m) gives, as a flat body facing
the camera: good enough for the frames and the fall, not for pointing towards the camera. Without
the calibration or the TF, there is no 3D: the people, postures and gestures are still published,
and a fall needs no height.

## The model

`yolo11n-pose.pt` (Ultralytics, AGPL-3.0, an optional dependency as for the `yolo` detector), real
time on a CPU (about 60 ms per image with two threads). The weights are downloaded once to
`~/.cache/perception_system` (or `$PERCEPTION_MODELS_DIR`) when the node is first configured;
`model_id` names other weights (`yolo11s-pose.pt`, a file), `device` the torch device.

## Limits

- The gestures and postures are geometric rules on 2D keypoints: they are fast and explainable,
  and as good as the keypoints. A person partly out of the image, seen from behind or from above
  (a camera on the ceiling) gives fewer keypoints, and `unknown` or nothing.
- Pointing at the camera foreshortens the arm in the image: it is seen with a depth image (the
  rule is then in 3D), not without one.
- A fall is a person lying on the floor for a second, not the movement of falling: someone who
  lies down on the floor on purpose is fallen too. For an alarm, confirm it (ask the person, wait
  longer).
- The ids are those of the body detector's tracker: they last while the track does, and a person
  who leaves and comes back gets another id.

# The markers

The skill `markers` is the lifecycle node `/marker_detector` of `perception_markers`
(`markers:=true`, the default), activated on request as the other skills. It finds fiducial
markers, ArUco or AprilTag, with OpenCV (no learned model, milliseconds per image) and their 6D
pose: a marker printed on a door, a charger, a shelf or an object tells the robot where it is and
how it is turned, to the centimetre.

| Output | Type | Content |
|---|---|---|
| `/markers/detections` | `vision_msgs/Detection3DArray` | One detection per marker, in the frame of the image: its id in `results[0].hypothesis.class_id` and `id`, its pose (z out of the marker) in `bbox.center` and `results[0].pose`, its size |
| `/markers/image` | `sensor_msgs/Image` | The markers outlined, with their ids and distances (only while someone subscribes) |
| `/tf` | | `marker_<id>`, the pose of each marker, child of `fixed_frame_id` (odom) |

The parameters ([CONFIGURATION.md](CONFIGURATION.md#marker_detector)): `dictionary`
(`DICT_4X4_50` by default; `DICT_APRILTAG_36h11` and the other dictionaries of OpenCV),
`marker_size_m` (the side of the black square, 0.10 m) and `marker_sizes` (`"id:size_m"` for the
markers of another size). The pose needs the calibration of the camera (its `CameraInfo`, with its
distortion, or fixed intrinsics); the side of the printed marker must be measured, since the
distance is proportional to it.
