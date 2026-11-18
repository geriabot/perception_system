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

"""The checks of the running perception system against the ground truth of synthetic_camera.py:
several clients at once, the 3D placement, the target frames, DetectOnce, the skills, stopping and
the life cycle. Run it with the system started as in evaluation/README.md; it prints one line per
check and exits with 1 if any fails.

    python3 check_system.py [--report report.json] [--skills]
"""

import argparse
import json
import math
import sys
import threading
import time

from lifecycle_msgs.msg import Transition
from lifecycle_msgs.srv import ChangeState
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float64, String
import tf2_ros
from vision_msgs.msg import Detection2DArray, Detection3DArray

from perception_system.client import MODE_GROUP, MODE_SINGLE, OUTPUT_DETECTIONS_2D, \
    OUTPUT_DETECTIONS_3D, OUTPUT_DETECTIONS_IMAGE, PerceptionClient, PerceptionError
from perception_system.transforms import apply

TOLERANCE_M = 0.06          # 3D placement against the ground truth
BOX_CENTER_HEIGHT = 0.85    # the center of a box's image (synthetic_camera.py)


class Checker(Node):

    def __init__(self):
        super().__init__('perception_system_checker')
        self.group = ReentrantCallbackGroup()
        self.client = PerceptionClient(self, callback_group=self.group)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.lock = threading.Lock()
        self.messages = {}
        self.truth = {}
        self.subscriptions_ = []
        self.results = []
        self.create_subscription(String, '/synthetic/ground_truth', self.on_truth,
                                 QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE,
                                            durability=DurabilityPolicy.VOLATILE))

    # ------------------------------------------------------------------ recording
    def on_truth(self, msg):
        data = json.loads(msg.data)
        with self.lock:
            self.truth[tuple(data['stamp'])] = data

    def record(self, topic, msg_type):
        with self.lock:
            self.messages.setdefault(topic, [])

        def callback(msg, topic=topic):
            with self.lock:
                self.messages[topic].append((time.monotonic(), msg))
        self.subscriptions_.append(self.create_subscription(msg_type, topic, callback, 10,
                                                            callback_group=self.group))

    def recorded(self, topic, since=0.0):
        with self.lock:
            return [m for t, m in self.messages.get(topic, []) if t >= since]

    def check(self, name, ok, detail=''):
        self.results.append({'check': name, 'ok': bool(ok), 'detail': detail})
        print(f'{"PASS" if ok else "FAIL"}  {name}{"  -- " + detail if detail else ""}',
              flush=True)
        return ok

    # ------------------------------------------------------------------ helpers
    def truth_at(self, stamp):
        with self.lock:
            return self.truth.get((stamp.sec, stamp.nanosec))

    def in_odom(self, header, x, y, z):
        transform = self.tf_buffer.lookup_transform(
            'odom', header.frame_id, Time.from_msg(header.stamp),
            timeout=rclpy.duration.Duration(seconds=1.0))
        return apply(transform.transform, (x, y, z))

    def nearest_box(self, truth, point, color=None):
        best = None
        for box in truth['boxes']:
            if color and box['color'] != color:
                continue
            d = math.dist((box['x'], box['y'], BOX_CENTER_HEIGHT), point)
            if best is None or d < best[0]:
                best = (d, box['name'])
        return best

    def frame_position(self, frame, timeout=1.0):
        try:
            transform = self.tf_buffer.lookup_transform('odom', frame, Time(),
                                                        timeout=rclpy.duration.Duration(
                                                            seconds=timeout))
        except tf2_ros.TransformException:
            return None
        t = transform.transform.translation
        return (t.x, t.y, t.z), transform.header.stamp

    def change_state(self, transition):
        client = self.create_client(ChangeState, '/perception_system/change_state',
                                    callback_group=self.group)
        client.wait_for_service(timeout_sec=5.0)
        request = ChangeState.Request()
        request.transition.id = transition
        done = threading.Event()
        future = client.call_async(request)
        future.add_done_callback(lambda _: done.set())
        done.wait(30.0)
        return future.done() and future.result().success

    # ------------------------------------------------------------------ the checks
    def run(self, skills):
        status = None
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            try:
                status = self.client.status(timeout_s=2.0)
                if status.state == 'active':
                    break
            except PerceptionError:
                pass
            time.sleep(0.5)
        if not self.check('the system is active', status is not None and
                          status.state == 'active'):
            return

        # Three clients at once
        attention = self.client.start_detection('attention', 'red person', 'red',
                                                output_namespace='/')
        self.check('detection request at the root', attention.detections_2d_topic ==
                   '/detections_2d' and attention.detections_3d_topic == '/detections_3d' and
                   attention.detections_image_topic == '/detections_image', attention.message)
        blue = self.client.start_detection('blue_watch', 'blue', outputs=OUTPUT_DETECTIONS_2D |
                                           OUTPUT_DETECTIONS_3D, tracker='bytetrack')
        self.check('detection request in its namespace', blue.detections_2d_topic ==
                   '/perception_system/requests/blue_watch/detections_2d' and
                   not blue.detections_image_topic)
        images = self.client.start_detection('images_only', outputs=OUTPUT_DETECTIONS_IMAGE)
        refused = self.client.call('start_detection', type(self.client).SERVICES[
            'start_detection'].Request(name='intruder', class_name='red', output_namespace='/'),
            check=False)
        self.check('a second request on the same namespace is refused', not refused.success,
                   refused.message)
        for topic in (attention.detections_2d_topic, blue.detections_2d_topic):
            self.record(topic, Detection2DArray)
        for topic in (attention.detections_3d_topic, blue.detections_3d_topic):
            self.record(topic, Detection3DArray)
        self.record(attention.detections_image_topic, Image)
        self.record(images.detections_image_topic, Image)
        self.record('/perception_system/frame_time', Float32)
        time.sleep(6.0)

        red2d = self.recorded('/detections_2d')
        labels = {r.hypothesis.class_id for m in red2d for d in m.detections for r in d.results}
        self.check('attention: only its class, the category published as the class',
                   red2d and labels == {'red person'}, f'{len(red2d)} messages, {labels}')
        counts = [len(m.detections) for m in red2d[3:]]
        self.check('attention: the two red boxes in every image', counts and
                   all(c == 2 for c in counts), f'counts {sorted(set(counts))}')
        ids = [sorted(d.id for d in m.detections) for m in red2d[5:]]
        self.check('attention: tracker ids confirmed and stable', ids and
                   all(i == ids[0] for i in ids) and '' not in ids[0], f'{ids[:1]}')
        with_truth = [m for m in red2d if self.truth_at(m.header.stamp)]
        self.check('detections stamped with the time of their image',
                   len(with_truth) == len(red2d), f'{len(with_truth)}/{len(red2d)}')
        blue2d = self.recorded(blue.detections_2d_topic)
        blue_labels = {r.hypothesis.class_id for m in blue2d for d in m.detections
                       for r in d.results}
        self.check('blue_watch: only blue, isolated from attention', blue2d and
                   blue_labels == {'blue'}, f'{len(blue2d)} messages')
        blue_ids = sorted({d.id for m in blue2d for d in m.detections if d.id})
        self.check('blue_watch: its own tracker numbering (ByteTrack, from 1)',
                   blue_ids[:1] == ['1'], f'ids {blue_ids}')
        image_stamps = {(m.header.stamp.sec, m.header.stamp.nanosec)
                        for m in self.recorded(images.detections_image_topic)}
        self.check('images_only: images without a class', len(image_stamps) >= 5,
                   f'{len(image_stamps)} images')

        errors, details = [], []
        boxes_2d = {(m.header.stamp.sec, m.header.stamp.nanosec): {d.id: d for d in m.detections}
                    for m in red2d}
        for msg in self.recorded('/detections_3d'):
            truth = self.truth_at(msg.header.stamp)
            if truth is None:
                continue
            for d in msg.detections:
                box = boxes_2d.get((msg.header.stamp.sec, msg.header.stamp.nanosec), {}).get(d.id)
                if box is not None:
                    left = box.bbox.center.position.x - box.bbox.size_x / 2
                    right = box.bbox.center.position.x + box.bbox.size_x / 2
                    if left <= 1.0 or right >= 638.0:
                        continue   # cut by the image: its center is not the box's
                p = d.bbox.center.position
                point = self.in_odom(msg.header, p.x, p.y, p.z)
                error, name = self.nearest_box(truth, point, 'red')
                errors.append(error)
                details.append(f'{name} {error:.3f}')
        self.check('3D placement against the ground truth (depth camera)', errors and
                   max(errors) < TOLERANCE_M,
                   f'{len(errors)} detections (those cut by the image left out), max error '
                   f'{max(errors or [0]):.3f} m'
                   + ('' if not errors or max(errors) < TOLERANCE_M else
                      f'; {[x for x in details if float(x.split()[1]) >= TOLERANCE_M][:5]}'))

        # Target frames: the still red box, and the group of both red boxes
        last = red2d[-1]
        still_id = None
        truth = self.truth_at(last.header.stamp)
        for msg in reversed(self.recorded('/detections_3d')):
            if self.truth_at(msg.header.stamp):
                for d in msg.detections:
                    p = d.bbox.center.position
                    if self.nearest_box(self.truth_at(msg.header.stamp),
                                        self.in_odom(msg.header, p.x, p.y, p.z))[1] == 'still':
                        still_id = d.id
                break
        reply = self.client.start_target_frame('attention_target', 'attention',
                                               'attention_point_1', 'red person', still_id,
                                               mode=MODE_SINGLE)
        self.check('target frame request', reply.success, reply.message)
        self.record('/perception_system/frames/group/half_width', Float64)
        self.client.start_target_frame('group', 'attention', 'group_point', 'red person',
                                       n_detections=2, mode=MODE_GROUP)
        time.sleep(3.0)
        position = self.frame_position('attention_point_1')
        error = math.dist(position[0], (4.0, 0.8, BOX_CENTER_HEIGHT)) if position else None
        self.check('target frame at the still box (odom)', error is not None and
                   error < TOLERANCE_M, f'error {error:.3f} m' if error is not None else
                   'no frame')
        position = self.frame_position('group_point')
        truth = self.truth_at(self.recorded('/detections_2d')[-1].header.stamp) or truth
        walker = next(b for b in truth['boxes'] if b['name'] == 'walker')
        midpoint = ((walker['x'] + 4.0) / 2, (walker['y'] + 0.8) / 2, BOX_CENTER_HEIGHT)
        error = math.dist(position[0], midpoint) if position else None
        # The walker moves up to 0.2 m/s: the frame is that of the last image processed
        self.check('group frame at the midpoint of both red boxes', error is not None and
                   error < 0.1, f'error {error:.3f} m' if error is not None else 'no frame')
        widths = [m.data for m in self.recorded('/perception_system/frames/group/half_width')]
        self.check('group half width published', widths and 0.3 < widths[-1] < 3.0,
                   f'{widths[-1]:.2f} m' if widths else 'none')
        status = self.client.status()
        kinds = {r.name: (r.kind, r.ok) for r in status.requests}
        self.check('status lists the five requests, the frames seen', set(kinds) == {
            'attention', 'blue_watch', 'images_only', 'attention_target', 'group'} and
            kinds['attention_target'][1] and kinds['group'][1], str(sorted(kinds)))
        self.check('status reports the speed', status.images_per_second >= 0.0 and
                   bool(status.models), f'models {list(status.models)}')
        frame_times = [m.data for m in self.recorded('/perception_system/frame_time')]
        self.check('frame_time published', bool(frame_times),
                   f'median {sorted(frame_times)[len(frame_times) // 2]:.1f} ms'
                   if frame_times else '')

        once = self.client.detect_once('blue', timeout_s=5.0)
        self.check('DetectOnce: the blue box, no id, in 3D', len(once.detections_2d.detections)
                   == 1 and once.detections_2d.detections[0].id == '' and
                   len(once.detections_3d.detections) == 1, once.message)
        state = self.client.user_state()
        self.check('user state served', state.available is False)

        # Stopping
        self.client.stop('attention')
        time.sleep(1.0)
        since = time.monotonic()
        time.sleep(1.5)
        status = self.client.status()
        names = {r.name for r in status.requests}
        self.check('stopping a detection ends its target frames', names == {
            'blue_watch', 'images_only'}, str(sorted(names)))
        self.check('a stopped request publishes nothing more',
                   not self.recorded('/detections_2d', since) and
                   self.recorded(blue.detections_2d_topic, since))
        reply = self.client.stop('attention')
        self.check('stopping an unknown request is answered', not reply.success, reply.message)

        if skills:
            try:
                reply = self.client.start_skill('dialogue', 'lip_activity', timeout_s=60.0)
                self.record('/lip_activity', Float32)
                time.sleep(2.0)
                ratios = self.recorded('/lip_activity')
                status = self.client.status()
                self.check('skill lip_activity: its node activated on request',
                           'lip_activity' in status.active_skills and bool(ratios),
                           f'{reply.message}; {len(ratios)} lip ratios')
                self.client.stop('dialogue')
                time.sleep(1.0)
                since = time.monotonic()
                time.sleep(1.0)
                self.check('skill lip_activity: its node deactivated after the last request',
                           'lip_activity' not in self.client.status().active_skills and
                           not self.recorded('/lip_activity', since))
            except PerceptionError as exc:
                self.check('skill lip_activity', False, str(exc))

        # Life cycle: deactivating ends every request
        ok = self.change_state(Transition.TRANSITION_DEACTIVATE)
        status = self.client.status()
        self.check('deactivation ends every request', ok and status.state == 'inactive' and
                   not status.requests, f'{status.state}, {len(status.requests)} requests')
        ok = self.change_state(Transition.TRANSITION_ACTIVATE)
        reply = self.client.start_detection('again', 'red')
        self.check('reactivation serves requests again', ok and reply.success)
        self.client.stop('again')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--report', default='', help='write the results to this JSON file')
    parser.add_argument('--skills', action='store_true',
                        help='also check the skill lip_activity (launch with user:=true)')
    args, _ = parser.parse_known_args()
    rclpy.init()
    node = Checker()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    try:
        node.run(args.skills)
    except Exception as exc:  # noqa: B902 - a crash is a failed check
        node.check('the checks ran to the end', False, f'{type(exc).__name__}: {exc}')
    failed = [r for r in node.results if not r['ok']]
    print(f'\n{len(node.results) - len(failed)}/{len(node.results)} checks passed', flush=True)
    if args.report:
        with open(args.report, 'w') as f:
            json.dump(node.results, f, indent=1)
    executor.shutdown()
    node.destroy_node()
    rclpy.try_shutdown()
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
