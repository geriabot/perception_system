# ROS interfaces

The types are in `perception_system_interfaces` (`msg/`, `srv/`, `action/`), whose files document
every field. Names below are the defaults; the namespaces of the requests and frames are
parameters ([CONFIGURATION.md](CONFIGURATION.md#perception_system)).

## Services of `/perception_system`

The request services exist while the node is active.

| Service | Type | What it does |
|---|---|---|
| `/perception_system/start_detection` | `StartDetection` | Opens a detection request, or updates the open one with the same name. Returns its topics |
| `/perception_system/start_target_frame` | `StartTargetFrame` | TF frames at the detections of a request: `MODE_SINGLE`, `MODE_GROUP`, `MODE_EACH`, `MODE_FOLLOW` (single, recognized again by its appearance when its id changes) |
| `/perception_system/start_skill` | `StartSkill` | Keeps a skill node active (`lip_activity`, `facial_expression`, `body`, `markers`). Returns its topics |
| `/perception_system/stop` | `StopRequest` | Ends a request by name (a detection request ends its frames too) |
| `/perception_system/detect_once` | `DetectOnce` | The detections of a class in the next image, without a request (no tracker ids) |
| `/perception_system/get_objects` | `GetObjects` | What a detection request saw in its last image: id, class, score, box, position in the fixed frame |
| `/perception_system/list_models` | `ListModels` | The models, the loaded ones, the default, and the classes of a closed-set model (loads it) |
| `/perception_system/get_status` | `GetStatus` | The status, as published (always available) |
| `/perception_system/get_user_state` | `GetUserState` | The latest state of the user: emotion and whether they speak (always available) |
| `/perception_system/change_state`, `get_state`, ... | `lifecycle_msgs` | Its life cycle |

A refused request answers `success: false` and the reason in `message`: an invalid name (a letter,
then letters, digits or `_`), a namespace or a TF frame used by another request, a name used by a
request of another kind, an unknown model, tracker or skill, a target frame without 3D placement.

## Topics of `/perception_system`

| Topic | Type | QoS | Content |
|---|---|---|---|
| `<namespace>/detections_2d` | `vision_msgs/Detection2DArray` | reliable, 10 | Boxes in pixels, class in `results[0].hypothesis.class_id`, tracker id in `id` (empty until confirmed), stamp and frame of the image |
| `<namespace>/detections_3d` | `vision_msgs/Detection3DArray` | reliable, 10 | The detections placed in 3D in the frame of the image, same ids; those with no estimate are left out |
| `<namespace>/detections_image` | `sensor_msgs/Image` (rgb8) | reliable, 10 | The image with every tracked detection boxed and labeled with its id (without a class: the image) |
| `<namespace>/debug_image` | `sensor_msgs/Image` (bgr8) | reliable, 10 | Every detection with class, id and score (`OUTPUT_DEBUG_IMAGE`) |
| `<namespace>/detection_centroid_markers` | `visualization_msgs/MarkerArray` | reliable, 10 | A sphere per 3D detection, for RViz |
| `/perception_system/frames/<name>/half_width` | `std_msgs/Float64` | reliable, 10 | `MODE_GROUP`: half width of the group across the line of sight plus half a body (m); the request may name another topic |
| `/perception_system/frames/<name>/tracked_ids` | `vision_msgs/Detection2DArray` | reliable, 10 | `MODE_GROUP`: the ids of the group (debug) |
| `/perception_system/frames/<name>/point` | `geometry_msgs/PointStamped` | reliable, 10 | `MODE_GROUP`: the midpoint in the fixed frame (debug) |
| `/perception_system/status` | `PerceptionStatus` | reliable, transient local, 1 | Life-cycle state, open requests (kind, description, outputs, whether a frame's target is seen), images/s, ms per image, failures, models, active skills; on every change and every `status_period_s` |
| `/perception_system/frame_time` | `std_msgs/Float32` | reliable, 10 | Milliseconds to process each image (every request, the network with the remote engine) |
| `/perception_system/frames/<name>/tracked_ids` | `vision_msgs/Detection2DArray` | reliable, 10 | `MODE_FOLLOW`: the current id of the followed object, on every image (empty while it is not seen) |
| `/tf` | `tf2_msgs/TFMessage` | — | The target frames, children of the fixed frame (`odom`), stamped with the time of the image |

`<namespace>` is `/perception_system/requests/<name>` by default; `output_namespace: "/"` puts the
topics at the root (`/detections_2d`, ...), where the attention system reads them.

Subscriptions: `image_topic` (`/image_rgb`, best effort, depth 1); with 3D placement
`projection.camera_info_topic` and, for `depth_image`, `projection.depth_image_topic`;
`/user_speaking` (`std_msgs/Bool`) and `/user_emotion` (`std_msgs/String`, JSON) for the user state.

## Actions of `/perception_orchestrator`

| Action | Type | Goal | Feedback | Result |
|---|---|---|---|---|
| `/perception_system/track_object` | `TrackObject` | class (and category), `selection` (`nearest`, `first`), `frame_id` (default `tracked_<class>`), `find_timeout_s` (0: no limit), `lost_timeout_s` (0: 2 s), `reacquire`, `model` | `state` (`SEARCHING`, `TRACKING`, `LOST`), `id`, the object (box, position in `odom`) | `reason` (`cancelled`, `not found`, `lost`), last `id` |
| `/perception_system/wait_for_object` | `WaitForObject` | class (and category), `min_count` (0: 1), `timeout_s` (0: no limit), `model` | `count` | `found`, `reason` (`found`, `timeout`, `cancelled`), the objects in view |
| `/perception_system/observe_class` | `ObserveClass` | class (and category) or `labels`, `frame_prefix` (default the class, or `object`), `model` | The tracked objects, with their positions | `reason` |
| `/perception_system/follow_person` | `FollowPerson` | class (default `person`, and category), `selection`, `frame_id` (default `followed_person`), `find_timeout_s` (0: no limit), `lost_timeout_s` (0: 10 s), `model` | `state` (`SEARCHING`, `FOLLOWING`, `LOST`), `id` (it changes when the person is recognized again), `reidentifications`, the person (box, position in `odom`) | `reason` (`cancelled`, `not found`, `lost`), last `id` |
| `/perception_system/wait_for_gesture` | `WaitForGesture` | `gesture` (`hand_raised`, `hand_raised_left`, `hand_raised_right`, `both_hands_raised`, `waving`, `pointing`, `pointing_left`, `pointing_right`; empty: any), `timeout_s` (0: no limit) | `people` in view | `found`, `reason` (`found`, `timeout`, `cancelled`), the `gesture` made and the `person` (`Person`: position, TF frame, place pointed at) |
| `/perception_system/measure_walk` | `MeasureWalk` | class (default `person`), `selection`, `frame_id` (default `walking_person`), `distance_m` (0 and `duration_s` 0: 4 m) or `duration_s`, `start_distance_m` (0: 0.2), `find_timeout_s`, `lost_timeout_s` (0: 3 s), `model`, `model_options`, `tracker` | `state` (`SEARCHING`, `WAITING_START`, `WALKING`), `id`, `distance_m`, `elapsed_s`, `speed_mps` | `reason` (`completed`, `not found`, `lost`, `cancelled`), `distance_m`, `time_s`, `speed_mps`, `start_stamp`, `end_stamp`, `id` |
| `/perception_system/watch_for_fall` | `WatchForFall` | `timeout_s` (0: no limit) | `people` in view | `fallen`, `reason` (`fallen`, `timeout`, `cancelled`), the fallen `person` |

The detection-based tasks (`track_object`, `follow_person`, `measure_walk`, `wait_for_object`,
`observe_class`) also take `model_options` and `tracker` (`strongsort`, `bytetrack`, `botsort`,
`ocsort`, `deepocsort`, `none`; empty: the default), as `start_detection` does.

A goal is refused when `/perception_system` is not active yet, when every runner of its behavior
is busy (`slots_per_behavior`), or when it is invalid (no class, negative times, an unknown
tracker or gesture). Every task runs
until it ends or is cancelled; its requests on `/perception_system` are named after its runner
(`perception_<behavior>_<n>`) and are stopped when it ends.

The runners publish on `/perception_system/behavior_status` (`std_msgs/String`), remapped from the
`/behavior_status` of `koch2`
([ARCHITECTURE.md](ARCHITECTURE.md#a-perception-task)).

## The skill nodes

| Node | Publishes | Subscribes |
|---|---|---|
| `/lip_activity_detector` | `/user_speaking` (`std_msgs/Bool`, on every change; `false` when deactivated while speaking), `/lip_activity` (`std_msgs/Float32`, lip gap ratio of each image) | `image_topic` |
| `/emotion_detector` | `/user_emotion` (`std_msgs/String`, JSON `{"emotion", "confidence", "source"}`) | `image_topic` |
| `/body_detector` | `/people/body` (`PersonArray`), `/people/events` (`BodyEvent`), `/people/body_image` (`sensor_msgs/Image`, while subscribed), `/tf` (`body_<id>`, `body_<id>_head`, `body_<id>_pointed`); serves `/body_detector/get_people` (`GetPeople`) | `image_topic`, `camera_info_topic`, `depth_image_topic` |
| `/marker_detector` | `/markers/detections` (`vision_msgs/Detection3DArray`), `/markers/image` (while subscribed), `/tf` (`marker_<id>`, with its orientation) | `image_topic`, `camera_info_topic` |

The messages of the body: `PersonArray` (header of the image, `fixed_frame_id`, `people`),
`Person` (id, score, box, 17 `BodyKeypoint`s, posture, `fallen`, gestures, `facing_camera`,
`clothing_color`, position and head in the fixed frame, pointing direction and place, TF frame),
`BodyEvent` (event, header, person). [BODY_PERCEPTION.md](BODY_PERCEPTION.md) explains them.

## The perception server

JSON over HTTP/1.1, documented in `perception_detectors/perception_detectors/protocol.py`:
`GET /health`, `POST /sessions`, `POST /sessions/<id>/class`, `POST /sessions/<id>/detect`,
`DELETE /sessions/<id>`, `POST /models` ([PERCEPTION_SERVER.md](PERCEPTION_SERVER.md)).
