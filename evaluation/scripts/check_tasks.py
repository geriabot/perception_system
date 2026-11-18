# Copyright 2026 Rodrigo Pérez-Rodríguez
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The checks of the perception tasks (the actions of /perception_orchestrator) and of the
services about objects and models, against the ground truth of synthetic_camera.py: TrackObject,
WaitForObject (found and timeout), ObserveClass (a frame per object), several tasks at once, a
full pool of slots, cancelling, GetObjects, ListModels and the frames of MODE_EACH. Run it with
the system started as in evaluation/README.md; one line per check, exit code 1 if any fails.

    python3 check_tasks.py [--report report.json]
"""

import argparse
import json
import math
import sys
import threading
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
import tf2_ros

from perception_system.client import MODE_EACH, PerceptionClient
from perception_system_interfaces.action import ObserveClass, TrackObject, WaitForObject
from perception_system_interfaces.srv import GetObjects, ListModels

TOLERANCE_M = 0.06
BOX_CENTER_HEIGHT = 0.85
SYSTEM = '/perception_system'


class TaskChecker(Node):

    def __init__(self):
        super().__init__('perception_task_checker')
        self.group = ReentrantCallbackGroup()
        self.client = PerceptionClient(self, callback_group=self.group)
        self.actions = {
            'track': ActionClient(self, TrackObject, f'{SYSTEM}/track_object',
                                  callback_group=self.group),
            'wait': ActionClient(self, WaitForObject, f'{SYSTEM}/wait_for_object',
                                 callback_group=self.group),
            'observe': ActionClient(self, ObserveClass, f'{SYSTEM}/observe_class',
                                    callback_group=self.group),
        }
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.truth = None
        self.lock = threading.Lock()
        self.results = []
        self.create_subscription(String, '/synthetic/ground_truth', self.on_truth, 10,
                                 callback_group=self.group)

    def on_truth(self, msg):
        with self.lock:
            self.truth = json.loads(msg.data)

    def box(self, name):
        with self.lock:
            return next(b for b in self.truth['boxes'] if b['name'] == name)

    def check(self, name, ok, detail=''):
        self.results.append({'check': name, 'ok': bool(ok), 'detail': detail})
        print(f'{"PASS" if ok else "FAIL"}  {name}{"  -- " + detail if detail else ""}',
              flush=True)
        return ok

    def wait(self, future, timeout=30.0):
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        done.wait(timeout)
        return future.result() if future.done() else None

    def send(self, action, goal, feedback=None):
        client = self.actions[action]
        client.wait_for_server(timeout_sec=30.0)
        handle = self.wait(client.send_goal_async(goal, feedback_callback=feedback))
        return handle

    def frame(self, name, timeout=2.0):
        try:
            t = self.tf_buffer.lookup_transform('odom', name, Time(),
                                                timeout=rclpy.duration.Duration(seconds=timeout))
        except tf2_ros.TransformException:
            return None
        p = t.transform.translation
        return (p.x, p.y, p.z), t.header.stamp

    def ground(self, name):
        b = self.box(name)
        return (b['x'], b['y'], BOX_CENTER_HEIGHT)

    def run(self):
        deadline = time.monotonic() + 90.0
        while time.monotonic() < deadline:
            try:
                if self.client.status(timeout_s=2.0).state == 'active' and \
                        self.actions['track'].wait_for_server(timeout_sec=1.0):
                    break
            except Exception:  # noqa: B902 - not up yet
                pass
            time.sleep(0.5)
        self.check('the orchestrator and /perception_system are up',
                   self.actions['track'].server_is_ready())
        time.sleep(2.0)   # the orchestrator accepts goals once it sees /perception_system active

        # ListModels
        models = self.client.call('list_models', ListModels.Request(model='color_blobs'))
        self.check('ListModels: the models and the classes of a closed-set one',
                   'yolo' in models.models and not models.open_vocabulary and
                   'red' in models.classes, f'{list(models.models)}')

        # TrackObject: the nearest red box is the walker. With reacquire, the task lives until it is
        # cancelled below: the walker is hidden or leaves the view at times, and without reacquire
        # the task would end as lost, depending on the moment of the scene
        feedbacks = []
        handle = self.send('track', TrackObject.Goal(
            class_name='red', find_timeout_s=10.0, lost_timeout_s=2.0, reacquire=True),
            lambda f: feedbacks.append(f.feedback))
        self.check('TrackObject accepted', handle is not None and handle.accepted)
        time.sleep(6.0)
        last = feedbacks[-1] if feedbacks else None
        self.check('TrackObject: feedback TRACKING with an id', last is not None and
                   last.state == 'TRACKING' and last.id != '',
                   f'{last.state} {last.id}' if last else 'no feedback')
        position = self.frame('tracked_red')
        error = math.dist(position[0], self.ground('walker')) if position else None
        self.check('TrackObject: the frame tracked_red on the nearest red box (the walker)',
                   error is not None and error < 0.15, f'error {error:.3f} m'
                   if error is not None else 'no frame')
        stamps_match = position is not None and position[1].sec > 0
        self.check('target frames carry the image stamp', stamps_match)
        obj = last.object if last else None
        self.check('TrackObject: feedback object with its position in odom',
                   obj is not None and obj.has_position and
                   obj.position.header.frame_id == 'odom')

        # Several tasks at once: WaitForObject found, and ObserveClass
        wait_feedback = []
        wait = self.send('wait', WaitForObject.Goal(class_name='red', min_count=2,
                                                    timeout_s=10.0),
                         lambda f: wait_feedback.append(f.feedback.count))
        observe_feedback = []
        observe = self.send('observe', ObserveClass.Goal(class_name='red'),
                            lambda f: observe_feedback.append(f.feedback))
        result = self.wait(wait.get_result_async(), 20.0)
        objects = result.result.objects if result else []
        # The still red box (never hidden): one of the objects, at its place
        errors = [math.dist((o.position.point.x, o.position.point.y, o.position.point.z),
                            self.ground('still')) for o in objects if o.has_position]
        error = min(errors) if errors else None
        self.check('WaitForObject: found the 2 red boxes, with their positions',
                   result is not None and result.result.found and len(objects) >= 2 and
                   error is not None and error < TOLERANCE_M,
                   f'{len(objects)} objects, still box error {error:.3f} m' if error is not None
                   else str(result.result.reason if result else 'no result'))
        time.sleep(4.0)
        ids = sorted(o.id for o in observe_feedback[-1].objects) if observe_feedback else []
        frames = {i: self.frame(f'red_{i}') for i in ids}
        self.check('ObserveClass: a frame red_<id> per red box, while tracking another',
                   len(ids) == 2 and all(frames.values()), f'ids {ids}')
        still = [f for f in frames.values() if f and math.dist(f[0], self.ground('still')) <
                 TOLERANCE_M]
        self.check('ObserveClass: one of the frames on the still box', len(still) == 1)
        status = self.client.status()
        names = sorted(r.name for r in status.requests)
        self.check('the tasks run on requests of /perception_system',
                   'perception_track_object_0' in names and 'perception_observe_class_0' in names,
                   str(names))

        # A full pool: slots_per_behavior (3) TrackObject goals at once, the fourth refused
        extra = [self.send('track', TrackObject.Goal(class_name='blue', find_timeout_s=10.0))
                 for _ in range(3)]
        accepted = [h is not None and h.accepted for h in extra]
        self.check('a full pool of slots refuses the goal', accepted == [True, True, False],
                   str(accepted))
        for h in extra:
            if h is not None and h.accepted:
                self.wait(h.cancel_goal_async())

        # WaitForObject that times out
        timeout = self.send('wait', WaitForObject.Goal(class_name='green', timeout_s=2.0))
        result = self.wait(timeout.get_result_async(), 10.0)
        self.check('WaitForObject: timeout', result is not None and not result.result.found and
                   result.result.reason == 'timeout',
                   result.result.reason if result else 'no result')

        # Cancelling ends the task and its requests
        self.wait(handle.cancel_goal_async())
        result = self.wait(handle.get_result_async(), 10.0)
        self.check('TrackObject: cancelled', result is not None and
                   result.result.reason == 'cancelled',
                   result.result.reason if result else 'no result')
        self.wait(observe.cancel_goal_async())
        self.wait(observe.get_result_async(), 10.0)
        time.sleep(1.5)
        names = sorted(r.name for r in self.client.status().requests)
        self.check('cancelled tasks leave no request open', not names, str(names))

        # Services: GetObjects and MODE_EACH directly
        self.client.start_detection('objects', 'red', outputs=3)
        self.client.start_target_frame('each_red', 'objects', 'red_box', '', mode=MODE_EACH)
        time.sleep(3.0)
        reply = self.client.call('get_objects', GetObjects.Request(request='objects',
                                                                   only_tracked=True))
        positions = [o for o in reply.objects if o.has_position]
        self.check('GetObjects: the tracked objects, with their position', len(reply.objects) ==
                   2 and len(positions) == 2, reply.message)
        frames = [self.frame(f'red_box_{o.id}') for o in reply.objects]
        self.check('MODE_EACH: a frame per object, <frame_id>_<id>', all(frames))
        self.client.stop('objects')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--report', default='')
    args, _ = parser.parse_known_args()
    rclpy.init()
    node = TaskChecker()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        node.run()
    except Exception as exc:  # noqa: B902 - a crash is a failed check
        import traceback
        traceback.print_exc()
        node.check('the checks ran to the end', False, f'{type(exc).__name__}: {exc}')
    failed = [r for r in node.results if not r['ok']]
    print(f'\n{len(node.results) - len(failed)}/{len(node.results)} checks passed', flush=True)
    if args.report:
        with open(args.report, 'w') as f:
            json.dump(node.results, f, indent=1)
    executor.shutdown()
    rclpy.try_shutdown()
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
