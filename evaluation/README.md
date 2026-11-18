# Evaluation

What is checked, why, and how to run it again. The results are written to `evaluation/results/`
(not versioned).

## What is checked

| Check | What it shows | Tool |
|---|---|---|
| Unit tests | The pipeline (classes, categories, trackers, marked image), the engines (requests isolated, one model run for equal labels, remote equal to local, a lost session reopened), the server (protocol, tokens, a persistent connection after an error), the target frames (`MODE_FOLLOW` recognizing its object again), the requests' rules, the 3D placement, the lip activity, the body (posture, fall, gestures, waving, pointing, facing, clothes, 3D placement on hand-made skeletons; the pose model on photographs of people standing and lying), the markers (pose of a drawn marker), and the Behavior Tree nodes against fake services | `pixi run test`, `colcon test` |
| The running system | Several clients at once (one at the root topics, one in its namespace, one with images only); classes, categories and ids isolated; stamps of the images; 3D placement and target frames (single, group, half width) against the ground truth; status; `detect_once`; the user state; stopping; the skill `lip_activity` activated and deactivated on request; deactivating and reactivating | `scripts/check_system.py` |
| The perception tasks | `TrackObject` (feedback, frame on the nearest object, image stamps, positions), `WaitForObject` (found with positions, timeout), `ObserveClass` (a frame per object), several tasks at once, a full pool of runners, cancelling, no request left open, `get_objects`, `list_models`, `MODE_EACH`, `FollowPerson` (the walker leaves the scene for longer than the tracker keeps its id and is recognized again with a new id; while it is away, the other red box is not taken), a task with the tracker chosen by the client and an unknown tracker refused, `MeasureWalk` (distance and speed of the walker against its ground truth between the image times of the start and the end) | `scripts/check_tasks.py` |
| The skills body and markers | With the pose model and photographs of people (the script is the camera): the people with their keypoints, standing, a TF frame per person and their head above it, the color of their clothes, `get_people`, `WaitForGesture` (timeout), a marker and its frame against its place, `WatchForFall` while the people fall (the image turns), the events `lying` and `fall`, the skill nodes deactivated after the last request | `scripts/check_skills.py` |
| Equivalence with the attention system | On the same images, `/perception_system` gives exactly what `attention_perception` gives | `equivalence/` |
| Equivalence with the interaction system | The lip activity gives the same state as `interaction_perception` at every image | `equivalence/compare_lip_activity.py` |

Unit tests also run every tracker (ids confirmed and stable, and each request numbering its own
tracks), and the Behavior Tree nodes `FrameInFront` and `MeasureDisplacement` against TF frames
published by the test. The 3D placement is checked on the boxes that the image does not cut (the
center of a cut box is not the box's).

The checks of the running system use a synthetic RGB-D camera (`scripts/synthetic_camera.py`):
a robot turning left and right in front of colored boxes at known places (a red one walking, a
red one still, a blue one), with its depth image, calibration and TF. The `color_blobs` model
detects the boxes by their color, so the expected outputs are exact and nothing needs a GPU or a
simulator. The 3D positions and the target frames are compared with the places of the boxes.

## Running the checks

On a ROS domain that nothing else uses, with the workspace built:

```sh
pixi shell
ROS_DOMAIN_ID=<free> CUDA_VISIBLE_DEVICES= bash src/perception_system/evaluation/scripts/run_check.sh \
  [<output folder>] tracker:=bytetrack
```

It starts the synthetic camera and the perception system (`color_blobs` on the CPU), runs
`check_tasks.py` and `check_system.py`, prints one line per check and writes `report*.json` and
the logs. `CHECK_TASKS=0` skips the tasks (with `orchestrator:=false`), `CHECK_SKILLS=0` the skill
`lip_activity` (an environment without MediaPipe). Further arguments go to the launch file, e.g. a
robot without PyTorch and a perception server:

```sh
python3 -m perception_detectors.perception_server --port 8769 --device cpu --preload color_blobs &
CHECK_TASKS=0 CHECK_SKILLS=0 bash .../run_check.sh <folder> orchestrator:=false engine:=remote \
  server_url:=http://localhost:8769
```

The skills body and markers have their own run, with a camera of their own (`check_skills.py`
publishes photographs of Ultralytics, a marker and the TF of a still camera; it needs
Ultralytics and the pose weights):

```sh
ROS_DOMAIN_ID=<free> CUDA_VISIBLE_DEVICES= bash src/perception_system/evaluation/scripts/run_skills_check.sh \
  [<output folder>]
```

## Equivalence with the attention system

`equivalence/run_equivalence.sh` runs, at the same time and on the same images of the synthetic
camera, the perception of the attention system (`attention_perception`: `/detector` with its
parameters, `detection_3d_estimator`, the two TF publishers) and `/perception_system`, gives both
the same request (a class and its category, StrongSORT), then a single target frame and a group
frame, and compares every output image by image: the detections (boxes, scores, classes, tracker
ids), the bytes of the marked images, the 3D detections, the positions of the target frames and
the half widths of the group.

```sh
# attention_perception, from the attention system's repository, in a folder of its own
mkdir -p legacy/src && git -C <attention_system> archive main \
  attention_perception/attention_perception_msgs attention_perception/attention_perception_nodes \
  attention_perception/attention_detectors | tar -x -C legacy/src
colcon build --symlink-install --base-paths legacy/src --build-base legacy/build \
  --install-base legacy/install
ROS_DOMAIN_ID=<free> CUDA_VISIBLE_DEVICES= LEGACY_INSTALL=$PWD/legacy/install \
  bash src/perception_system/evaluation/equivalence/run_equivalence.sh [<output folder>]
```

`legacy_detector.py` adds the `color_blobs` model to the detector of the attention system without
changing it. The camera publishes 320x240 images, and only after both systems have the request,
so that both trackers start at the same image. The verdict is `EQUIVALENT`, `DIFFERENT`, or
`INCONCLUSIVE` when the two systems did not process the same images (DDS may drop a best-effort
image for one subscriber): one image less changes a tracker's state from there on, so such a run
shows nothing, and the script repeats it (up to `ATTEMPTS`, 3). The positions of the target
frames are compared, not their stamps: the perception system stamps them with the time of the
image, `attention_perception` with the time of publication.

## Equivalence with the interaction system

```sh
python3 src/perception_system/evaluation/equivalence/compare_lip_activity.py \
  <interaction_system>/interaction_perception/interaction_perception/interaction_perception
```

It feeds the lip-activity classification of both (the original with its ROS and MediaPipe
imports replaced) the same random sequences of lip-gap ratios (silence, speech, single openings,
missing faces) at 10, 15 and 30 images per second, with the same clock, and counts the images
where the speaking state differs.

## The tools

| Tool | What it does |
|---|---|
| `scripts/synthetic_camera.py` | The synthetic RGB-D camera, with its ground truth on `/synthetic/ground_truth` |
| `scripts/image_camera.py` | Publishes an image file again and again on `/image_rgb`, to try a real model without a robot |
| `scripts/face_camera.py` | A drawn face on `/image_rgb`, silent or talking (`/face_camera/talking`, `std_msgs/Bool`): the lip activity (skill `lip_activity`) without a camera or a person. The interaction system's integration test uses it |
| `scripts/check_system.py`, `scripts/check_tasks.py`, `scripts/run_check.sh` | The checks of the running system (`/synthetic/hidden` makes a box of the synthetic camera leave the scene) |
| `scripts/check_skills.py`, `scripts/run_skills_check.sh` | The checks of the skills body and markers |
| `equivalence/compare_with_attention.py`, `legacy_detector.py`, `run_equivalence.sh` | The equivalence with the attention system |
| `equivalence/compare_lip_activity.py` | The equivalence with the interaction system |
