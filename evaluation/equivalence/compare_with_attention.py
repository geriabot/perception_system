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

"""The perception system against the perception of the attention system (attention_perception:
/detector, detection_3d_estimator and the two TF publishers), on the same images of the synthetic
camera, at the same time. Both receive the same request (a class and its category), then a single
target frame and a group frame; every output is compared image by image:

  detections_2d     boxes, scores, classes and tracker ids, exactly
  detections_image  the pixels of the marked image, exactly
  detections_3d     ids, classes and positions, exactly
  target frames     the positions published in odom (old: attention_old_*, new: attention_new_*)
  half width        the half width of the group

Started by run_equivalence.sh (the camera runs inside this process, from the moment both
systems have the request, so that both trackers start at the same image).
"""

import argparse
import json
import math
import os
import sys
import threading
import time

from attention_perception_msgs.srv import SetDetectionClass, StartNDetectionsSameClassTFPub, \
    StartSingleDetectionTFPub
from lifecycle_msgs.msg import Transition
from lifecycle_msgs.srv import ChangeState
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from tf2_msgs.msg import TFMessage
from vision_msgs.msg import Detection2DArray, Detection3DArray

from perception_system.client import MODE_GROUP, MODE_SINGLE, PerceptionClient

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from synthetic_camera import set_resolution, SyntheticCamera  # noqa: E402

NEW = '/perception_system/requests/attention'
TOPICS = {
    'old_2d': ('/detections_2d', Detection2DArray), 'new_2d': (f'{NEW}/detections_2d',
                                                                Detection2DArray),
    'old_3d': ('/detections_3d', Detection3DArray), 'new_3d': (f'{NEW}/detections_3d',
                                                                Detection3DArray),
    'old_image': ('/detections_image', Image), 'new_image': (f'{NEW}/detections_image', Image),
    'old_half_width': ('/attention/target_half_width', Float64),
    'new_half_width': ('/perception_system/frames/group/half_width', Float64),
}


def stamp_of(msg):
    return msg.header.stamp.sec, msg.header.stamp.nanosec


def rows_2d(msg):
    return [(d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.size_x, d.bbox.size_y,
             d.results[0].hypothesis.score, d.results[0].hypothesis.class_id, d.id)
            for d in msg.detections]


def rows_3d(msg):
    return [(d.id, d.results[0].hypothesis.class_id, d.bbox.center.position.x,
             d.bbox.center.position.y, d.bbox.center.position.z, d.bbox.size.x, d.bbox.size.y)
            for d in msg.detections]


class Comparison(Node):

    def __init__(self):
        super().__init__('perception_equivalence')
        self.group = ReentrantCallbackGroup()
        self.lock = threading.Lock()
        self.data = {key: [] for key in TOPICS}
        self.frames = {}          # child frame -> [(x, y, z)]
        for key, (topic, msg_type) in TOPICS.items():
            self.create_subscription(msg_type, topic, lambda m, k=key: self.store(k, m), 50,
                                     callback_group=self.group)
        self.create_subscription(TFMessage, '/tf', self.on_tf, 100, callback_group=self.group)
        self.client = PerceptionClient(self, callback_group=self.group)

    def store(self, key, msg):
        with self.lock:
            self.data[key].append(msg)

    def on_tf(self, msg):
        for t in msg.transforms:
            if t.child_frame_id.startswith('attention_'):
                p = t.transform.translation
                with self.lock:
                    self.frames.setdefault(t.child_frame_id, []).append((p.x, p.y, p.z))

    def call(self, srv_type, name, request, timeout=30.0):
        client = self.create_client(srv_type, name, callback_group=self.group)
        if not client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f'{name} is not available')
        done = threading.Event()
        future = client.call_async(request)
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout):
            raise RuntimeError(f'No response from {name}')
        return future.result()

    def setup(self, class_name, category):
        for transition in (Transition.TRANSITION_CONFIGURE, Transition.TRANSITION_ACTIVATE):
            request = ChangeState.Request()
            request.transition.id = transition
            if not self.call(ChangeState, '/detector/change_state', request, 120.0).success:
                raise RuntimeError('The detector of the attention system did not start')
        self.call(SetDetectionClass, '/detection_class',
                  SetDetectionClass.Request(class_name=class_name, category=category))
        deadline = time.monotonic() + 120.0
        while self.client.status().state != 'active':
            if time.monotonic() > deadline:
                raise RuntimeError('The perception system did not become active')
            time.sleep(0.5)
        self.client.start_detection('attention', class_name, category)

    def start_frames(self, class_name, single_id):
        self.call(StartSingleDetectionTFPub, '/start_single_detection_tf_publisher',
                  StartSingleDetectionTFPub.Request(frame_id='attention_old_single',
                                                    id=single_id, class_name=class_name))
        self.call(StartNDetectionsSameClassTFPub, '/start_n_detections_same_class_tf_publisher',
                  StartNDetectionsSameClassTFPub.Request(frame_id='attention_old_group',
                                                         n_detections=2, class_name=class_name))
        self.client.start_target_frame('single', 'attention', 'attention_new_single', class_name,
                                       single_id, mode=MODE_SINGLE)
        self.client.start_target_frame('group', 'attention', 'attention_new_group', class_name,
                                       n_detections=2, mode=MODE_GROUP)

    def compare(self):
        with self.lock:
            data = {k: list(v) for k, v in self.data.items()}
            frames = {k: list(v) for k, v in self.frames.items()}
        report = {}
        for kind, rows in (('2d', rows_2d), ('3d', rows_3d)):
            old = {stamp_of(m): rows(m) for m in data[f'old_{kind}']}
            new = {stamp_of(m): rows(m) for m in data[f'new_{kind}']}
            common = sorted(set(old) & set(new))
            differ = [s for s in common if old[s] != new[s]]
            report[f'detections_{kind}'] = {
                'old_messages': len(old), 'new_messages': len(new), 'common_images': len(common),
                'different': len(differ), 'detections': sum(len(old[s]) for s in common),
                'first_difference': None if not differ else
                {'stamp': differ[0], 'old': old[differ[0]], 'new': new[differ[0]]}}
        old = {stamp_of(m): bytes(m.data) for m in data['old_image']}
        new = {stamp_of(m): bytes(m.data) for m in data['new_image']}
        common = set(old) & set(new)
        report['detections_image'] = {'common_images': len(common),
                                      'different': sum(old[s] != new[s] for s in common)}
        for kind in ('single', 'group'):
            old = frames.get(f'attention_old_{kind}', [])
            new = frames.get(f'attention_new_{kind}', [])
            matched = 0
            pool = list(new)
            for p in old:
                best = min(pool, key=lambda q: math.dist(p, q), default=None)
                if best is not None and math.dist(p, best) < 1e-9:
                    matched += 1
                    pool.remove(best)
            report[f'frame_{kind}'] = {'old': len(old), 'new': len(new),
                                       'old_found_in_new': matched}
        old = [m.data for m in data['old_half_width']]
        new = [m.data for m in data['new_half_width']]
        report['half_width'] = {'old': len(old), 'new': len(new),
                                'equal_sequence': old == new}
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--class-name', default='red person')
    parser.add_argument('--category', default='red')
    parser.add_argument('--rate', type=float, default=1.0, help='images per second')
    parser.add_argument('--before-frames', type=float, default=8.0, help='s of images first')
    parser.add_argument('--after-frames', type=float, default=20.0, help='s with the frames')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='image size of the camera (0.5: 320x240, smaller messages for DDS)')
    parser.add_argument('--report', default='equivalence.json')
    args, _ = parser.parse_known_args()
    set_resolution(args.scale)
    rclpy.init()
    node = Comparison()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    node.setup(args.class_name, args.category)
    print('Both systems have the request: the camera starts', flush=True)
    camera = SyntheticCamera(args.rate, 0.15, 0.0, warmup=3.0)
    executor.add_node(camera)
    time.sleep(3.0 + args.before_frames)
    # The single target: the still red box, the farthest (optical frame: z forward), by the id
    # that both give it (the ids are compared too)
    with node.lock:
        last = node.data['old_3d'][-1]
    single = max(last.detections, key=lambda d: d.bbox.center.position.z)
    node.start_frames(args.class_name, single.id)
    print(f'Target frames started (single: id {single.id})', flush=True)
    time.sleep(args.after_frames)
    executor.remove_node(camera)
    time.sleep(3.0)
    report = node.compare()
    with open(args.report, 'w') as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    # The comparison holds only if both processed the same images: a lost image (best-effort
    # QoS) changes the tracker's state from there on. Then the run says nothing: repeat it
    same_input = all(report[k]['old_messages'] == report[k]['new_messages'] ==
                     report[k]['common_images'] for k in ('detections_2d', 'detections_3d'))
    ok = all(report[k]['different'] == 0 and report[k]['common_images'] > 0
             for k in ('detections_2d', 'detections_3d', 'detections_image')) and \
        all(report[k]['old'] > 0 and report[k]['old_found_in_new'] >= report[k]['old'] - 1
            for k in ('frame_single', 'frame_group'))
    verdict = 'EQUIVALENT' if ok else 'DIFFERENT' if same_input else \
        'INCONCLUSIVE (the systems did not process the same images: repeat the run)'
    print(verdict, flush=True)
    executor.shutdown()
    rclpy.try_shutdown()
    sys.exit(0 if ok else 1 if same_input else 2)


if __name__ == '__main__':
    main()
