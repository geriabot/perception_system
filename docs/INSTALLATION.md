# Installation

## The sources

```sh
mkdir -p ~/perception_ws/src && cd ~/perception_ws/src
git clone https://github.com/geriabot/perception_system.git
vcs import < perception_system/thirdparty.repos     # detection_3d_estimators, koch2
```

## With pixi (any Linux x86-64)

[pixi](https://pixi.sh) installs ROS 2 (RoboStack) and every dependency in the workspace, without
a system ROS. The manifest is in the repository; a script links it at the root of the workspace,
where the environments (`.pixi/`) live:

```sh
cd ~/perception_ws
./src/perception_system/pixi/setup_workspace.sh
pixi run build              # ROS 2 Jazzy: build/ and install/
pixi shell                  # a shell with the environment and the workspace
pixi run test               # the unit tests
```

| Environment | ROS 2 | Builds | Notes |
|---|---|---|---|
| `default` | Jazzy | Everything | PyTorch (CUDA 12.6 wheels, also on a CPU), Ultralytics, MediaPipe |
| `lyrical` | Lyrical | Everything but the orchestrator: `pixi run -e lyrical build-lyrical` (`build_lyrical/`, `install_lyrical/`) | No PyTorch: use `engine:=remote` (a perception server) and `orchestrator:=false`, until `koch2` builds on Lyrical |

The environment sets `PYTHONNOUSERSITE=1` (the packages of `~/.local` must not mix with its own)
and keeps its fontconfig cache apart from the host's (`pixi/activate-fontconfig.sh`: rviz2 may
crash otherwise).

## Native (system ROS 2 Jazzy, Ubuntu 24.04)

Not verified on a clean computer; the pixi environment is the reference.

```sh
sudo apt install ros-dev-tools
sudo rosdep init && rosdep update        # only the first time
cd ~/perception_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
pip install --break-system-packages -r src/perception_system/requirements.txt
colcon build --symlink-install \
  --packages-up-to perception_system perception_orchestrator perception_system_behaviors \
    perception_user
source src/perception_system/activate.sh     # in every terminal (edit it once)
```

With a virtual environment instead of `--break-system-packages`, create it with
`--system-site-packages` (rclpy comes from the system), install `requirements.txt` in it and set
`PERCEPTION_VENV` in `activate.sh`.

## Models

The model weights are downloaded the first time they are used: Hugging Face models to
`~/.cache/huggingface`, Ultralytics and the re-identification model of StrongSORT to the working
directory, MediaPipe's to `~/.cache/perception_system`. A robot without internet needs them copied
beforehand. SAM 3 needs access to its gated weights
([PERCEPTION_SERVER.md](PERCEPTION_SERVER.md#sam-3)).
