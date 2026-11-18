# Integrating the perception system

A client chooses how to use it ([CATALOG.md](CATALOG.md) lists what to use for each need):

| A client that... | Uses |
|---|---|
| needs a task done (follow an object or a person, wait for one, frames on every object of a class, wait for a gesture, watch for a fall) | the actions of `/perception_orchestrator` |
| has its own logic and consumes perception continuously | detection, target frame and skill requests on `/perception_system` |
| asks a single question | `detect_once`, `get_objects` |
| is a Behavior Tree | the plugin `perception_system_bt_nodes`, or the perception behaviors as subtrees ([BEHAVIOR_TREES.md](BEHAVIOR_TREES.md)) |

Rules for every client:

- **Names.** A request name identifies the request for every client: prefix it with the client
  (`attention`, `dialogue_lips`, `fetch_cup`). The same name updates the request; another client's
  name would take it over.
- **End what you open.** A request stays open until `stop` (or the deactivation of the system);
  an action ends its requests itself. A Behavior Tree ends its requests in its cleanup branch.
- **Namespaces and frames are exclusive.** Two requests cannot publish on the same namespace, nor
  two target frame requests the same frame.
- **Stamps.** Detections and target frames carry the time of the image: combine them with the
  robot's pose at that time (`lookupTransform` at the stamp), or with the latest one when the
  delay does not matter.
- **Launching it from another launch file.** Launch arguments are shared by every launch file
  included in the same launch: an `image_topic`, `params_file` or `model` of another one (the
  TurtleBot 4's declares `image_topic`) would reach `perception_system.launch.py`. Include it in a
  scoped group without forwarding, with its arguments explicit:
  `GroupAction([IncludeLaunchDescription(...)], scoped=True, forwarding=False)`.

## From a behavior of the robot

In a mission of `koch2`, list the plugin and use the nodes:

```yaml
plugin_libraries:
  - "libperception_system_bt_nodes.so"
```

```xml
<Sequence>
  <StartDetection request="fetch_cup" class="cup" detections_2d_topic="{cups}"/>
  <RetryUntilSuccessful num_attempts="50">
    <Sequence>
      <Sleep msec="100"/>
      <DetectionSeen topic="{cups}" class="cup"/>
    </Sequence>
  </RetryUntilSuccessful>
  <StopPerception request="fetch_cup"/>
</Sequence>
```

Or include a perception behavior as a subtree
(`<include ros_pkg="perception_system_behaviors" path="behavior_tree_xml/track_object.xml"/>`),
or call the actions from any node. A behavior that reacts to people uses the skill `body`:

```xml
<Sequence>
  <!-- Someone waves at the robot: go to them -->
  <StartPerceptionSkill request="call_robot" skill="body"/>
  <RetryUntilSuccessful num_attempts="-1">
    <Sequence>
      <Sleep msec="100"/>
      <GestureSeen gesture="waving" person_frame="{caller}"/>
    </Sequence>
  </RetryUntilSuccessful>
  <StopPerception request="call_robot"/>
  <!-- ... navigate to {caller} ... -->
</Sequence>
```

## From Python

```python
from perception_system.client import PerceptionClient, MODE_EACH

client = PerceptionClient(node)            # call from a thread of its own (see client.py)
reply = client.start_detection('inventory', labels=['chair', 'table', 'tv'], model='yolo')
client.start_target_frame('inventory_frames', 'inventory', 'furniture', '', mode=MODE_EACH)
objects = client.objects('inventory', only_tracked=True).objects   # ids, classes, odom positions
client.stop('inventory')

client.start_skill('carer_body', 'body')   # then /people/body, /people/events, the body_<id> frames
```

## The attention system

The attention system used its own perception, `attention_perception`, through the interface listed
in its README. The perception system serves that interface with the same topics and the same
behavior: on the same images, its detections (boxes, scores, classes, tracker ids), marked
images, 3D detections and target frames are equal to those of `attention_perception`
([evaluation](../evaluation/README.md#equivalence-with-the-attention-system)). Only the request
services change, and the target frames carry the time of the image instead of the time of
publication.

### What changes

| `attention_perception` | `perception_system` |
|---|---|
| `attention_perception.launch.py` | `perception_system.launch.py startup_requests:=attention:/` |
| lifecycle node `/detector`, activated at start | `/perception_system`, activated by the orchestrator (or `autostart`); with `startup_requests` it publishes the camera image on `/detections_image` from activation, as `/detector` did without a class |
| `/detection_class` (`SetDetectionClass`: class, category; empty: stop) | `/perception_system/start_detection` (`StartDetection`: `name: attention`, the class and category, `output_namespace: "/"`); an empty class publishes the images only |
| `/start_single_detection_tf_publisher` (`StartSingleDetectionTFPub`: frame, id, class) | `/perception_system/start_target_frame` (`StartTargetFrame`: `name: attention_target`, `detection_request: attention`, `MODE_SINGLE`, frame, class, id) |
| `/stop_single_detection_tf_publisher` (`Trigger`) | `/perception_system/stop` (`StopRequest`: `attention_target`) |
| `/start_n_detections_same_class_tf_publisher` (frame, n, class) | `start_target_frame` with `MODE_GROUP`, `n_detections`, `half_width_topic: /attention/target_half_width` |
| `/stop_n_detections_same_class_tf_publisher` | `stop` (`attention_group`) |
| `/detector/frame_time` | `/perception_system/frame_time` |
| `attention_perception_msgs` | `perception_system_interfaces` |

Unchanged: `/detections_2d`, `/detections_image`, `/detections_3d`,
`/detection_centroid_markers`, `/attention/target_half_width`, the frame names, the fixed frame
(`odom`), the image topic (`/image_rgb`) and the trackers' parameters.

The launch arguments map one to one: `detector:=remote` → `engine:=remote`,
`detector_model` → `model`, `detector_model_options` → `model_options`, `detector_device` →
`device`, `detector_tracker` → `tracker`, `detector_zoom` → `zoom_small_detections`,
`perception_server_url` → `server_url`, `perception_server_token_env` → `token_env`,
`perception_server_timeout_s` → `server_timeout_s`, `projector_mode:=rgbd|astra|no_depth_cam` →
`camera:=rgbd|astra|nao` (`rgbd_body` when `camera_frame_is_optical` was `false`), and
`depth_estimator`, `camera_intrinsics`, `start_perception_server` as they are.
`object_sizes`, `estimate_classes`, `unknown_distance_m` and `detector_added_latency_ms` go in a
`params_file` (`projection.sizes`, `projection.classes`, `projection.unknown_distance_m`,
`remote.added_latency_ms`). `detector_debug` is per request (`OUTPUT_DEBUG_IMAGE`).

### The steps

1. Add `perception_system` to the attention system's `thirdparty.repos`, and remove
   `attention_perception/` (and `detection_3d_estimators`, which comes with it).
2. In `attention_system_behaviors`, replace `attention_perception_msgs` by
   `perception_system_interfaces`, and make the four nodes that call its services use the new
   ones (table above): `ChangeVisualDetectionClass` and `ClearVisualDetectionClass` call
   `start_detection` with the name `attention` and `output_namespace: "/"`;
   `RequestSingleDetectionTrackPoint` and `RequestNDetectionsSameClassTrackPoint` call
   `start_target_frame`, and their cleanup calls `stop`. The `service_name` attributes of the
   trees change accordingly. The nodes that only read topics do not change.
3. Start `perception_system.launch.py` with `startup_requests:=attention:/` and the camera of the
   robot instead of `attention_perception.launch.py`.
4. The actuations that use the age of the target frame (the frame now has the time of the image,
   not of publication) take the detector's latency into account.

## The interaction system

The interaction system used `interaction_perception` through a topic and a service:

| `interaction_perception` | `perception_system` |
|---|---|
| `lip_activity_detector` (always processing) | `/lip_activity_detector`, active while the skill `lip_activity` is requested: `start_skill` (`name: dialogue_lips`, `skill: lip_activity`), or `autostart_skills:=lip_activity` |
| `/user_speaking` (`std_msgs/Bool`), `/lip_activity` | the same topics, the same classification |
| `/interaction_perception/get_user_state` (`interaction_perception_msgs/srv/GetUserState`) | `/perception_system/get_user_state` (`perception_system_interfaces/srv/GetUserState`, the same fields) |
| `emotion_detector` | `/emotion_detector`, skill `facial_expression` (experimental) |
| `interaction_perception.launch.py` | `perception_system.launch.py user:=true` |

The steps: depend on `perception_system_interfaces` instead of `interaction_perception_msgs` in
`interaction_intelligence`, set `user_state_service` to `/perception_system/get_user_state`,
start the perception system with `user:=true autostart_skills:=lip_activity` (or request the skill when
a dialogue starts and stop it when it ends, so that the lips are processed only while talking),
and remove `interaction_perception/`.

## A new capability or subsystem

Use the actions for tasks, the requests for streams, prefix every request name with the
subsystem, and stop the requests in the subsystem's deactivation. If its orchestrator is built
with `koch2`, give its runners names unique in the robot (the perception system
uses `perception_<behavior>_<n>`).
