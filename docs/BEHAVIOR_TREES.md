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
| `StartTargetFrame` | Action | `/perception_system/start_target_frame` | `request`, `detection_request`, `frame_id`, `mode` (`single`, `group`, `each`), `class` (required except for `each`), `id`, `n_detections`, `fixed_frame_id`, `half_width_topic` |
| `StartPerceptionSkill` | Action | `/perception_system/start_skill` | `request`, `skill` |
| `StopPerception` | Action | `/perception_system/stop` | `request` |
| `DetectOnce` | Action | `/perception_system/detect_once` | `class`, `category`, `labels`, `model`, `min_count` (1); out: `count`. `SUCCESS` if at least `min_count` |
| `DetectionSeen` | Condition | — | `topic` (a `detections_2d` topic), `class`, `id`, `min_count` (1), `max_age_ms` (1000); out: `count`. `SUCCESS` if the latest message, at most `max_age_ms` old, has at least `min_count` matching detections |
| `SelectDetection` | Condition | — | `topic` (a `detections_3d` topic), `class`, `policy` (`nearest`, `first`), `max_age_ms`; out: `id`. Chooses a tracked detection of the class |

## The perception behaviors

In `behavior_tree_xml/`, following the conventions of the reusable trees of
`koch2`: each file starts with a comment with its plugin, needs, ports, outputs,
result and usage.

| Tree | What it does | Result |
|---|---|---|
| `TrackObject` (`track_object.xml`) | Opens a detection request, waits for a tracked object of the class (`SelectDetection`), publishes a frame on it, and keeps checking it. Lost for `lost_timeout_ms`: with `reacquire`, another object of the class; else it ends | Runs while tracking; `FAILURE` if none is found within `find_timeout_ms` or it is lost |
| `WaitForObject` (`wait_for_object.xml`) | Opens a detection request and waits for `min_count` objects of the class | `SUCCESS` when they are in view; `FAILURE` after `timeout_ms` |
| `ObserveClass` (`observe_class.xml`) | Opens a detection request and a `MODE_EACH` frame request | Runs until halted |

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

`find_timeout_ms` and `lost_timeout_ms` are read by `Timeout` as `unsigned`: given from a
blackboard entry, it must be of that type (the orchestrator writes them so).

## How the orchestrator runs them

Each behavior has `slots_per_behavior` runners, named `perception_<behavior>_<n>`. For each goal,
the orchestrator gives a free runner a tree that includes the behavior as a subtree with the
goal's values, and its outputs in keys of the runner; it activates the runner, reads its status
and outputs every control cycle, and deactivates it and stops its requests when the goal ends
([ARCHITECTURE.md](ARCHITECTURE.md#a-perception-task)). A node that throws while the tree is
ticked stops the whole `mission_executor` process, as in any runner of `koch2`:
the trees use only the nodes above, with typed ports.
