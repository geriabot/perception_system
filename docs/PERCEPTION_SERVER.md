# The perception server

`perception_server` runs the detection models for the remote engine of one or more robots. It
needs no ROS: only Python and `perception_detectors/requirements-server.txt`. Each detection
request of a robot is a session of the server, with its own class and tracker; the models are
shared between sessions. Use it when the robot's computer has no GPU, or not enough for the
model, or no PyTorch (the remote engine of the robot imports no PyTorch).

```text
robot:   camera ─▶ /perception_system (remote engine) ──JPEG over HTTP──▶ perception_server
                          ▲                                                 (models, trackers)
                          └──────────── detections with ids ◀───────────────┘
```

## 1. Install (on the GPU computer)

```sh
git clone https://github.com/geriabot/perception_system.git    # only perception_detectors is needed
cd perception_system/perception_detectors
python3 -m venv ~/perception_venv && source ~/perception_venv/bin/activate   # Python 3.10-3.12
pip install -r requirements-server.txt
pip install ultralytics                                         # for yolo and yolo_world
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The `torch` of `requirements-server.txt` comes with CUDA 12.6; with an older driver, install the
`torch` that matches it first (pytorch.org).

### SAM 3

SAM 3 needs Python 3.12 or later, its own package and access to its weights:

1. Request access on <https://huggingface.co/facebook/sam3>.
2. In a separate environment:
   ```sh
   python3.12 -m venv ~/sam3_venv && source ~/sam3_venv/bin/activate
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   git clone https://github.com/facebookresearch/sam3.git && pip install -e sam3
   pip install transformers==4.56.2 boxmot==16.0.11 "opencv-python-headless<5" pyyaml
   hf auth login
   ```
3. Start the server from that environment with `--preload sam3`.

## 2. Run

```sh
cd perception_system/perception_detectors && source ~/perception_venv/bin/activate
python3 -m perception_detectors.perception_server --port 8765 --preload omdet
```

| Option | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | `0.0.0.0` listens on the network (use `--token`) |
| `--port` | `8765` | |
| `--device` | `cuda` if available | `cuda:1`, `cpu` |
| `--token` | | Every request needs `Authorization: Bearer <token>` |
| `--preload` | | Models loaded at start; the others on their first session |

Sessions without requests for 10 minutes are closed; the robot then opens a new one and sets its
class again. To keep it running after closing the terminal, use `tmux`, `nohup` or a systemd
service.

## 3. Connect the robot

Through an SSH tunnel (the images travel encrypted):

```sh
ssh -N -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -L 8765:localhost:8765 <user>@<server>
curl http://localhost:8765/health
ros2 launch perception_system perception_system.launch.py engine:=remote model:=grounding_dino ...
```

Or on the network: `--host 0.0.0.0 --token <secret>` on the server, and on the robot
`export PERCEPTION_TOKEN=<secret>` and `engine:=remote server_url:=http://<server>:8765
token_env:=PERCEPTION_TOKEN`.

## 4. Check

- `curl http://localhost:8765/health`: the models, the loaded ones, the device, the sessions.
- The log of `/perception_system` gives the server and its models on activation, and every
  `stats_period_s` the images per second and milliseconds per image, network included
  (`/perception_system/frame_time`).
- `Cannot reach the perception server`: the tunnel is closed or the server is not running.
- With `image_format: png` the detections equal those of the local engine; with JPEG (the default,
  smaller) scores and boxes may change slightly.

When the network fails, an image without an answer within `remote.timeout_s` publishes no
detections: the clients see their targets as lost until the server answers again.

The protocol is in `perception_detectors/perception_detectors/protocol.py`.
