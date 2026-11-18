# Configuration

The launch file loads, in this order: `perception_system_core/perception_system/config/
perception_system.yaml` (every parameter with its default), `config/cameras/<camera>.yaml`, the
file of `params_file`, and the launch arguments that are not empty
([LAUNCH_ARGUMENTS.md](LAUNCH_ARGUMENTS.md)).

## `/perception_system`

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `image_topic` | string | `/image_rgb` | The camera image |
| `engine` | string | `local` | `local`: the models on this computer; `remote`: in a perception server (`remote.*`) |
| `model` | string | `omdet` | Default model of the requests ([MODELS.md](MODELS.md)) |
| `model_options` | string | `""` | Its options, YAML (`"{threshold: 0.35}"`) |
| `device` | string | `""` | Torch device of the models and the tracker; `""`: `cuda` if available |
| `preload_model` | bool | `true` | Load the default model on activation |
| `tracker` | string | `strongsort` | Default tracker of the requests: `strongsort`, `bytetrack` or `none` |
| `zoom_small_detections` | bool | `false` | Add to `detections_image` a strip with the small detections enlarged (low-resolution cameras) |
| `default_namespace` | string | `/perception_system/requests` | Namespace of the topics of a request without `output_namespace` |
| `frames_namespace` | string | `/perception_system/frames` | Namespace of the topics of the target frames |
| `fixed_frame_id` | string | `odom` | Default parent of the target frames, and frame of the positions of `get_objects` |
| `tf_lookup_timeout_ms` | int | `1000` | How long to wait for the camera pose at the time of an image |
| `frame_period_ms` | int | `100` | Period of each target frame's publisher (a frame is published once per new image) |
| `skill_names`, `skill_nodes`, `skill_topics` | string[] | `speaking, emotion`; `lip_activity_detector, emotion_detector`; `"/user_speaking,/lip_activity", "/user_emotion"` | The skills: name, lifecycle node, topics (comma-separated) |
| `autostart_skills` | string[] | `[""]` | Skills active from activation |
| `skill_timeout_s` | double | `10.0` | Timeout of the life-cycle calls to a skill node |
| `user_speaking_topic`, `user_emotion_topic` | string | `/user_speaking`, `/user_emotion` | Read for `get_user_state` |
| `startup_requests` | string[] | `[""]` | Detection requests opened on activation, without a class: `name` or `name:output_namespace` (`attention:/`) |
| `status_period_s` | double | `1.0` | Period of `/perception_system/status` |
| `stats_period_s` | double | `30.0` | Period of the speed in the log and the status |
| `autostart` | bool | `false` | Configure and activate itself at start (the launch file sets it when there is no orchestrator) |

### `tracking.*`: StrongSORT and ByteTrack

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `max_age` | int | `45` | Images a track survives without a detection (ByteTrack: its buffer) |
| `min_hits` | int | `3` | StrongSORT: detections before a track is confirmed (gets an id) |
| `iou_threshold` | double | `0.3` | Minimum IoU to give a track's id to a detection |
| `min_conf` | double | `0.25` | Minimum score of a detection (ByteTrack: its track threshold) |
| `max_cos_dist`, `max_iou_dist`, `nn_budget`, `mc_lambda`, `ema_alpha` | double/int | `0.25`, `0.875`, `80`, `0.98`, `0.9` | StrongSORT's association |
| `reid_weights` | string | `osnet_x0_25_msmt17.pt` | StrongSORT's re-identification model (downloaded by boxmot the first time) |

These are the values of the attention system. StrongSORT re-identifies by appearance and keeps
ids through occlusions and crossings; ByteTrack is far cheaper on a CPU.

### `remote.*`: the remote engine

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `server_url` | string | `http://localhost:8765` | The perception server (through an SSH tunnel: `localhost`) |
| `token_env` | string | `""` | Environment variable with the server's token |
| `image_format` | string | `jpeg` | `jpeg` or `png` (lossless: the same detections as the local engine) |
| `jpeg_quality` | int | `90` | |
| `timeout_s` | double | `2.0` | Per image; an image without an answer publishes no detections |
| `wait_for_server_s` | double | `60.0` | How long activation waits for the server |
| `connect_timeout_s` | double | `600.0` | To open a session (the server may load the model) |
| `added_latency_ms` | double | `0.0` | Tests: delay added to every image |

### `projection.*`: the 3D placement

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `estimator` | string | `none` | `none` (no 3D), `depth_image`, `ray`, `box_size`, `depth_anything` |
| `camera_info_topic` | string | `/camera_info` | The calibration |
| `camera_intrinsics` | double[] | `[0.0]` | `[fx, fy, cx, cy]` instead of the `CameraInfo` |
| `camera_frame_is_optical` | bool | `true` | `false` when the images are stamped with a body frame (x forward, y left, z up) |
| `camera_frame_id` | string | `""` | Frame of the images that have none |
| `depth_image_topic` | string | `/depth_image` | `depth_image`: the depth (16UC1 mm or 32FC1 m), aligned with the color image |
| `depth_window_px`, `depth_region`, `depth_percentile` | int, double, double | `5`, `0.6`, `10.0` | `depth_image`, `depth_anything`: the depth of a box is a low percentile of the central part of it (its nearest surface) |
| `ray_distance_m` | double | `1.0` | `ray`: the distance of every detection |
| `object_sizes_file`, `sizes`, `classes`, `cut_margin` | string, string[], string[], double | the package's table, `[""]`, `[""]`, `0.025` | `box_size`: the table of real sizes, sizes added (`name:height[:width]`), only these classes, margin of a box cut by the image |
| `unknown_distance_m` | double | `1.0` | `box_size`, `depth_anything`: the distance of what has no estimate |
| `depth_anything_model`, `depth_anything_device` | string | `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf`, `cuda` | `depth_anything` |
| `marker_scale`, `marker_color` | double, double[] | `0.05`, `[0, 1, 0, 1]` | The RViz spheres |

The estimators are those of
[`detection_3d_estimators`](https://github.com/rodperex/detection_3d_estimators#readme), whose
README explains each one.

### Cameras

`config/cameras/<camera>.yaml`, chosen with the launch argument `camera`:

| Camera | Robots | Placement |
|---|---|---|
| `none` | Any | No 3D (no `detections_3d`, no target frames) |
| `rgbd` | TurtleBot 4, real or simulated | `depth_image` from `/rgbd_camera/depth_image` and `/rgbd_camera/camera_info`, optical frame |
| `rgbd_body` | Unitree R1, simulated Kobuki | As `rgbd`, images stamped with a body frame |
| `astra` | Kobuki with an Orbbec Astra | `depth_image` from `/depth/image_raw` and `/color/camera_info` |
| `nao` | NAO | `ray` (1 m) from `/camera_rgb_info`, body frame `CameraTop_frame` |

A robot of your own: a YAML file with the same structure, given with `params_file`.

## `/perception_orchestrator`

Parameters of the orchestrator node (the launch file passes them to `mission_executor`):

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `slots_per_behavior` | int | `3` | Runners of each perception behavior: goals of a task at the same time |
| `perception_system` | string | `/perception_system` | The sensing node: services, actions and life cycle |
| `manage_perception_system` | bool | `true` | Configure and activate it in cascade |
| `feedback_period_ms` | int | `500` | Period of the feedback of the actions |
| `tick_period_ms` | int | `50` | Tick period of the trees |
| `plugin_library` | string | `libperception_system_bt_nodes.so` | The BT nodes of the trees |
| `control_cycle_rate_ms` | int | `100` | Of `BaseOrchestrator`: period of the control cycle |

The configuration of `mission_executor` is `perception_orchestrator/config/perception_mission.yaml`.

## The skill nodes

`/lip_activity_detector`: `image_topic` (`/image_rgb`), `speaking_topic` (`/user_speaking`),
`lip_activity_topic` (`/lip_activity`), `min_detection_confidence` and `min_tracking_confidence`
(0.5, MediaPipe), `smoothing_frames` (4), `roughness_threshold` (7e-6), `roughness_window` (10),
`onset_secs` (0.9), `silence_secs` (1.2). How they act: [USER_PERCEPTION.md](USER_PERCEPTION.md).

`/emotion_detector`: `image_topic`, `output_topic` (`/user_emotion`), `model_name`
(`dima806/face_emotions_image_detection`), `min_detection_confidence` (0.5).
