# The catalog: what the perception system offers

What a behavior or another capability of the robot can ask the perception system, by what it
needs. Each row gives the interface to use: a **task** (an action of the orchestrator, which ends
by itself), a **request** (a service that opens a stream of topics and frames, until `stop`), a
**skill** (a node activated with `start_skill`) or a **question** (a service answered once).
[ROS_INTERFACES.md](ROS_INTERFACES.md) has every type, [INTEGRATING.md](INTEGRATING.md) how to use
them from a tree or from Python.

## Objects

| I need | Use | What I get |
|---|---|---|
| To know whether something is in view now | `detect_once` (question) | The detections of a class in the next image, in 2D and 3D |
| To wait until something appears ("is someone at the door?") | `WaitForObject` (task) | `found` and the objects, or `timeout` |
| The detections of a class in every image | `start_detection` (request) | `detections_2d`, `detections_3d`, `detections_image`, with tracker ids |
| A TF frame on one object | `start_target_frame` `MODE_SINGLE` (request), or `TrackObject` (task) | The frame in `odom` while it is seen; the task also searches, follows and reports `LOST` |
| A frame on every object of a class (every chair of the room) | `start_target_frame` `MODE_EACH`, or `ObserveClass` (task) | `<prefix>_<id>` per object |
| A frame on the middle of a group, and its width | `start_target_frame` `MODE_GROUP` (request) | The frame and `half_width` (what the attention system looks at) |
| What a request sees now, with positions | `get_objects` (question) | id, class, score, box, position in `odom` |
| The models and their classes | `list_models` (question) | |
| A marker on a door, a charger or a shelf, and its pose | skill `markers` | `/markers/detections`, the frame `marker_<id>` with its orientation ([BODY_PERCEPTION.md](BODY_PERCEPTION.md#the-markers)) |

## People

| I need | Use | What I get |
|---|---|---|
| To follow a person, as when the robot accompanies them | `FollowPerson` (task) | The frame `followed_person` on them; when the tracker loses them (they leave, someone crosses), they are recognized again by their appearance and never confused with another person seen with them; `LOST` while they are away |
| To follow anyone of a class, taking another one if lost | `TrackObject` with `reacquire` (task) | The frame on whoever of the class is nearest |
| How fast a person walks (a gait speed test) | `MeasureWalk` (task) | Distance, time and speed from when they start walking, with the times of the images; the behavior gives the instructions |
| How far, or how fast, anything with a TF frame moves | `MeasureDisplacement` (BT node) on a target frame | Distance on the floor, time, speed |
| To turn the robot until it faces someone | `FrameInFront` (BT node) on their frame | Whether they are in front, and the side to turn to |
| The people in view, with their bodies | skill `body` | `/people/body`: keypoints, posture, gestures, where they point, facing the camera, clothes; the frames `body_<id>`, `body_<id>_head` ([BODY_PERCEPTION.md](BODY_PERCEPTION.md)) |
| To know when someone calls the robot (waves, raises a hand) | `WaitForGesture` (task) | Who, the gesture, their frame and position |
| Where a person points on the floor ("bring me that") | `WaitForGesture` with `pointing` (task), or skill `body` | `pointed_location` and the frame `body_<id>_pointed` |
| To raise an alarm when someone falls | `WatchForFall` (task), or the events of the skill `body` | The fallen person, their position and frame; the events `fall` and `recovered` |
| Whether someone sits down, stands up, lies down | the events of the skill `body` | `/people/events`: `sitting`, `standing`, `lying` |
| Who looks at the robot (who wants to talk) | skill `body` | `facing_camera` of each person |
| "The person in red" without a language model | skill `body`, or a request with an open-vocabulary model | `clothing_color`; or the class `person in red` with the category `person` |
| A crowd: how many, and a frame on each | `WaitForObject` with `min_count`, `ObserveClass` with `person` | |

## The user (in front of the robot)

| I need | Use | What I get |
|---|---|---|
| Whether the user is speaking, from the movement of their lips (visual barge-in) | skill `lip_activity` | `/user_speaking` ([USER_PERCEPTION.md](USER_PERCEPTION.md)) |
| Their facial expression (experimental) | skill `facial_expression` | `/user_emotion` |
| Both, once | `get_user_state` (question) | |

## Choosing the model and the tracker

Every detection request and every detection-based task names its own `model` (`omdet`,
`grounding_dino`, `yolo`, `yolo_world`, `sam3`, `color_blobs`), `model_options` and `tracker`
(`strongsort`, `bytetrack`, `botsort`, `ocsort`, `deepocsort`, `none`); empty, the defaults of the
system. [MODELS.md](MODELS.md) says which suits what: YOLO and ByteTrack or BoT-SORT on a CPU, the
80 classes of COCO; OMDet-Turbo or Grounding DINO for any text; StrongSORT or Deep OC-SORT when
ids must survive crossings.

## Choosing

- A task when the behavior wants something done and a result (found, lost, fallen); a request
  when it consumes a stream for as long as it likes; a skill when it reads what the skill
  publishes.
- Several clients use the same capability at once: each request and each task has its own
  outputs; a skill node is shared (one node, the same topics for every client) and stays active
  while any client needs it.
- The tasks are also subtrees that a behavior of the robot can include in its own tree
  ([BEHAVIOR_TREES.md](BEHAVIOR_TREES.md)).
