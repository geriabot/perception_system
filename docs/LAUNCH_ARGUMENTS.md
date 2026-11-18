# Launch arguments

`ros2 launch perception_system perception_system.launch.py <argument>:=<value>`. An empty
argument keeps the value of the configuration files ([CONFIGURATION.md](CONFIGURATION.md)), so
only what is given here overrides them.

| Argument | Default | Meaning | Parameter |
|---|---|---|---|
| `camera` | `none` | The camera of the robot: `rgbd`, `rgbd_body`, `astra`, `nao`, `none` | file `config/cameras/<camera>.yaml` |
| `params_file` | | A YAML file of parameters, loaded after the camera | |
| `image_topic` | | The camera image | `image_topic` |
| `engine` | | `local` or `remote` | `engine` |
| `model` | | Default detection model | `model` |
| `model_options` | | Its options, YAML | `model_options` |
| `device` | | Torch device (`cpu`, `cuda:1`) | `device` |
| `tracker` | | Default tracker: `strongsort`, `bytetrack`, `none` | `tracker` |
| `preload_model` | | `true` or `false` | `preload_model` |
| `zoom_small_detections` | | `true` or `false` | `zoom_small_detections` |
| `server_url` | | The perception server | `remote.server_url` |
| `token_env` | | Environment variable with its token | `remote.token_env` |
| `server_timeout_s` | | Maximum time per image | `remote.timeout_s` |
| `start_perception_server` | `false` | Also start a perception server on this computer, on the port of `server_url` (tests) | |
| `camera_intrinsics` | | `"[fx, fy, cx, cy]"` instead of the `CameraInfo` | `projection.camera_intrinsics` |
| `depth_estimator` | | The 3D placement, overriding the camera's | `projection.estimator` |
| `fixed_frame_id` | | Parent of the target frames | `fixed_frame_id` |
| `startup_requests` | | Requests opened on activation, `"name"` or `"name:namespace"`, comma-separated | `startup_requests` |
| `user` | `false` | Start `/lip_activity_detector` and `/emotion_detector` (unconfigured, activated on request) | |
| `autostart_skills` | | Skills active from activation, `"speaking,emotion"` | `autostart_skills` |
| `orchestrator` | `true` | Start `mission_executor` with `/perception_orchestrator` (the actions) | |
| `slots_per_behavior` | `3` | Goals of each task at the same time | `slots_per_behavior` (orchestrator) |
| `autostart` | | `/perception_system` activates itself; default `true` without the orchestrator, `false` with it (the orchestrator activates it) | `autostart` |
| `use_sim_time` | `false` | | `use_sim_time` of every node |
| `log_level` | `info` | | |

Examples:

```sh
# TurtleBot 4 in simulation, the model on this computer
ros2 launch perception_system perception_system.launch.py camera:=rgbd use_sim_time:=true

# NAO, the model on a GPU server through an SSH tunnel, the lips of the user for barge-in
ros2 launch perception_system perception_system.launch.py camera:=nao engine:=remote \
  model:=grounding_dino user:=true autostart_skills:=speaking

# A computer without a GPU: YOLO on the CPU, ByteTrack
ros2 launch perception_system perception_system.launch.py model:=yolo \
  model_options:="{model_id: yolo11n.pt}" device:=cpu tracker:=bytetrack
```
