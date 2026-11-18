# Behavior Trees: the nodes and the perception behaviors

The package `perception_system_behaviors` has the BehaviorTree.CPP 4 nodes of the perception
system, in one plugin library, `libperception_system_bt_nodes.so`, and the perception behaviors,
the trees that `/perception_orchestrator` runs for its actions. The nodes take the `rclcpp::Node`
of the tree from the blackboard entry `node` (or `@node` from a subtree), as the runners of
`koch2` set it.

## The nodes

Every action node calls its service without blocking the tick: it waits for the service and its
answer within `timeout_ms`, and returns `FAILURE` if the perception system refuses the request
(the reason is logged) or does not answer. Each has the ports `service_name` (the default below)
and `timeout_ms`.

| Node | Kind | Service (default) | Ports |
|---|---|---|---|
| `StartDetection` | Action | `/perception_system/start_detection` | `request`; `class`, `category`, `labels` (`"cup; bowl"`), `output_namespace`, `outputs` (int), `model`, `model_options`, `tracker`; out: `detections_2d_topic`, `detections_3d_topic`, `detections_image_topic` |
| `StartTargetFrame` | Action | `/perception_system/start_target_frame` | `request`, `detection_request`, `frame_id`, `mode` (`single`, `group`, `each`, `follow`), `class` (required except for `each`), `id`, `n_detections`, `fixed_frame_id`, `half_width_topic` |
| `StartPerceptionSkill` | Action | `/perception_system/start_skill` | `request`, `skill` (`lip_activity`, `facial_expression`, `body`, `markers`) |
| `StopPerception` | Action | `/perception_system/stop` | `request` |
| `DetectOnce` | Action | `/perception_system/detect_once` | `class`, `category`, `labels`, `model`, `min_count` (1); out: `count`. `SUCCESS` if at least `min_count` |
| `DetectionSeen` | Condition | — | `topic` (a `detections_2d` topic), `class`, `id`, `min_count` (1), `max_age_ms` (1000); out: `count`. `SUCCESS` if the latest message, at most `max_age_ms` old, has at least `min_count` matching detections |
| `SelectDetection` | Condition | — | `topic` (a `detections_3d` topic), `class`, `policy` (`nearest`, `first`), `max_distance` (m, 0: any), `max_age_ms`; out: `id`. Chooses a tracked detection of the class |
| `FrameTargetSeen` | Condition | — | `frame_request` (in `MODE_FOLLOW`), `frames_namespace` (`/perception_system/frames`), `max_age_ms`; out: `id`. `SUCCESS` while the followed object is seen; `id` is its current tracker id |
| `GestureSeen` | Condition | — | `gesture` (`hand_raised`, `hand_raised_left`, `hand_raised_right`, `both_hands_raised`, `waving`, `pointing`, `pointing_left`, `pointing_right`; empty: any), `topic` (`/people/body`), `max_age_ms`; out: `person_id`, `person_frame`, `made`, `people`. `SUCCESS` if someone makes the gesture (needs the skill `body`) |
| `PersonFallen` | Condition | — | `topic` (`/people/body`), `max_age_ms`; out: `person_id`, `person_frame`, `people`. `SUCCESS` if someone is fallen (needs the skill `body`) |
| `FrameInFront` | Condition | — | `frame`, `reference_frame` (`base_link`), `max_angle_deg` (5); out: `bearing` (rad, positive to the left), `direction` (1 left, -1 right, 0 in front), `distance`. `SUCCESS` if the frame is in front of the robot: with a node that turns the robot, it faces someone (as `IsInFront` of `bt_nodes`) |
| `MeasureDisplacement` | Action | — | `frame`, `fixed_frame` (`odom`), `start_distance` (0.2 m), `distance` (m, 0: no limit), `duration_ms` (0: no limit), `lost_timeout_ms` (3000); out: `state` (`WAITING_START`, `WALKING`), `walked`, `elapsed`, `speed`, `start_time`, `end_time`. Waits until the frame moves `start_distance`, then measures from there, with the times of the TF frames (those of the images); `SUCCESS` at the distance or the duration, `FAILURE` if the frame is not updated for `lost_timeout_ms` (as `HasPersonStarted` and `CheckEndTest` of `gait_speed_ros2`) |

## The perception behaviors

In `behavior_tree_xml/`, following the conventions of the reusable trees of
`koch2`: each file starts with a comment with its plugin, needs, ports, outputs,
result and usage.

| Tree | What it does | Result |
|---|---|---|
| `TrackObject` (`track_object.xml`) | Opens a detection request, waits for a tracked object of the class (`SelectDetection`), publishes a frame on it, and keeps checking it. Lost for `lost_timeout_ms`: with `reacquire`, another object of the class; else it ends | Runs while tracking; `FAILURE` if none is found within `find_timeout_ms` or it is lost |
| `WaitForObject` (`wait_for_object.xml`) | Opens a detection request and waits for `min_count` objects of the class | `SUCCESS` when they are in view; `FAILURE` after `timeout_ms` |
| `ObserveClass` (`observe_class.xml`) | Opens a detection request and a `MODE_EACH` frame request | Runs until halted |
| `MeasureWalk` (`measure_walk.xml`) | As `FollowPerson` until the frame is published, then `MeasureDisplacement` on it | `SUCCESS` when the person has walked the distance (or the time has passed); `FAILURE` if nobody is found, or they are lost |
| `FollowPerson` (`follow_person.xml`) | Opens a detection request, chooses a tracked person (`SelectDetection`), publishes a `MODE_FOLLOW` frame on them (recognized again by their appearance when their id changes), and keeps checking it (`FrameTargetSeen`) | Runs while following; `FAILURE` if nobody is found within `find_timeout_ms`, or they are not seen for `lost_timeout_ms` |
| `WaitForGesture` (`wait_for_gesture.xml`) | Requests the skill `body` and waits for a gesture (`GestureSeen`) | `SUCCESS` with who made it; `FAILURE` after `timeout_ms` |
| `WatchForFall` (`watch_for_fall.xml`) | Requests the skill `body` and waits until someone is fallen (`PersonFallen`) | `SUCCESS` with who; `FAILURE` after `timeout_ms` |

The trees leave their requests open: whoever runs them ends them with `StopPerception` (the
orchestrator does it when a goal ends). Used as a subtree in a behavior of the robot:

```xml
<root BTCPP_format="4">
  <include ros_pkg="perception_system_behaviors" path="behavior_tree_xml/track_object.xml"/>
  <BehaviorTree ID="GreetSomeone">
    <Sequence>
      <SubTree ID="TrackObject" request="greet" frame_request="greet_frame" class="person"
               category="" model="" frame_id="person_to_greet" selection="nearest"
               find_timeout_ms="20000" lost_timeout_ms="2000" reacquire="false"
               id="{person}" state="{tracking}"/>
      <!-- ... -->
    </Sequence>
  </BehaviorTree>
</root>
```

Every tree that opens a detection request takes the ports `model`, `model_options` and `tracker`
(empty: the defaults of the system). `find_timeout_ms`, `lost_timeout_ms`, `timeout_ms` and
`duration_ms` are read as `unsigned`: given from a
blackboard entry, it must be of that type (the orchestrator writes them so).

## How the orchestrator runs them

Each behavior has `slots_per_behavior` runners, named `perception_<behavior>_<n>`. For each goal,
the orchestrator gives a free runner a tree that includes the behavior as a subtree with the
goal's values, and its outputs in keys of the runner; it activates the runner, reads its status
and outputs every control cycle, and deactivates it and stops its requests when the goal ends
([ARCHITECTURE.md](ARCHITECTURE.md#a-perception-task)). A node that throws while the tree is
ticked fails its behavior: `koch2` halts the tree and ends it with `FAILURE`, and the other
runners of `mission_executor` go on (the goal then ends as a failed task).
