# Index

Every command of the repository, with the document that explains it. Commands run from the root
of the workspace, in `pixi shell` (or a terminal prepared with `activate.sh`).

## Installing and building

| Command | Document |
|---|---|
| `vcs import < src/perception_system/thirdparty.repos` (from `src/`) | [INSTALLATION.md](docs/INSTALLATION.md#the-sources) |
| `./src/perception_system/pixi/setup_workspace.sh` | [INSTALLATION.md](docs/INSTALLATION.md#with-pixi-any-linux-x86-64) |
| `pixi run build`, `pixi run -e lyrical build-lyrical`, `pixi shell` | [INSTALLATION.md](docs/INSTALLATION.md#with-pixi-any-linux-x86-64) |
| `pip install -r src/perception_system/requirements.txt`, `source .../activate.sh` | [INSTALLATION.md](docs/INSTALLATION.md#native-system-ros-2-jazzy-ubuntu-2404) |

## Running

| Command | Document |
|---|---|
| `ros2 launch perception_system perception_system.launch.py camera:=<camera> ...` | [LAUNCH_ARGUMENTS.md](docs/LAUNCH_ARGUMENTS.md) |
| `python3 -m perception_detectors.perception_server --port 8765 --preload omdet` | [PERCEPTION_SERVER.md](docs/PERCEPTION_SERVER.md) |
| `ssh -N -L 8765:localhost:8765 <user>@<server>`; `curl http://localhost:8765/health` | [PERCEPTION_SERVER.md](docs/PERCEPTION_SERVER.md#3-connect-the-robot) |

## Using it

| Command | Document |
|---|---|
| `ros2 service call /perception_system/start_detection ...`, `start_target_frame`, `start_skill`, `stop` | [ROS_INTERFACES.md](docs/ROS_INTERFACES.md#services-of-perception_system) |
| `ros2 service call /perception_system/get_objects ...`, `detect_once`, `list_models`, `get_status`, `get_user_state` | [ROS_INTERFACES.md](docs/ROS_INTERFACES.md#services-of-perception_system) |
| `ros2 topic echo /perception_system/status` | [ROS_INTERFACES.md](docs/ROS_INTERFACES.md#topics-of-perception_system) |
| `ros2 action send_goal /perception_system/track_object ...`, `wait_for_object`, `observe_class`, `follow_person`, `measure_walk`, `wait_for_gesture`, `watch_for_fall` | [ROS_INTERFACES.md](docs/ROS_INTERFACES.md#actions-of-perception_orchestrator) |
| What to use for each need (objects, people, the user) | [CATALOG.md](docs/CATALOG.md) |
| `ros2 topic echo /people/body`, `/people/events`, `/markers/detections`; `ros2 service call /body_detector/get_people ...` | [BODY_PERCEPTION.md](docs/BODY_PERCEPTION.md) |
| Behavior Tree nodes and perception behaviors | [BEHAVIOR_TREES.md](docs/BEHAVIOR_TREES.md) |
| `PerceptionClient` (Python) | [INTEGRATING.md](docs/INTEGRATING.md#from-python) |

## Testing and validating

| Command | Document |
|---|---|
| `pixi run test` | [README](README.md#tests) |
| `colcon test --packages-select perception_system_behaviors` | [README](README.md#tests) |
| `bash src/perception_system/evaluation/scripts/run_check.sh` | [evaluation/README.md](evaluation/README.md#running-the-checks) |
| `bash src/perception_system/evaluation/scripts/run_skills_check.sh` | [evaluation/README.md](evaluation/README.md#running-the-checks) |
| `bash src/perception_system/evaluation/equivalence/run_equivalence.sh` | [evaluation/README.md](evaluation/README.md#equivalence-with-the-attention-system) |
| `python3 src/perception_system/evaluation/equivalence/compare_lip_activity.py <path>` | [evaluation/README.md](evaluation/README.md#equivalence-with-the-interaction-system) |
| `python3 src/perception_system/evaluation/scripts/image_camera.py <image>` | [evaluation/README.md](evaluation/README.md#the-tools) |
| `python3 src/perception_system/evaluation/scripts/face_camera.py [--talking]` | [evaluation/README.md](evaluation/README.md#the-tools) |

## Reading order

1. [README](README.md): what it is and what it offers; [CATALOG.md](docs/CATALOG.md): what to use
   for each need.
2. [ARCHITECTURE.md](docs/ARCHITECTURE.md): how it works and why.
3. [INTEGRATING.md](docs/INTEGRATING.md): using it from a client.
4. **Reference**: [ROS_INTERFACES.md](docs/ROS_INTERFACES.md),
   [CONFIGURATION.md](docs/CONFIGURATION.md), [LAUNCH_ARGUMENTS.md](docs/LAUNCH_ARGUMENTS.md), [BEHAVIOR_TREES.md](docs/BEHAVIOR_TREES.md),
   [MODELS.md](docs/MODELS.md), [PERCEPTION_SERVER.md](docs/PERCEPTION_SERVER.md),
   [USER_PERCEPTION.md](docs/USER_PERCEPTION.md), [BODY_PERCEPTION.md](docs/BODY_PERCEPTION.md).
