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

"""The checks of the skills body and markers, with a pose model and real photographs: this script
is the camera. A still camera 0.9 m over the floor looking ahead (odom -> camera_optical,
calibration on /camera_info) shows the people of an image of Ultralytics standing (bus.jpg), with
an ArUco marker (DICT_4X4_50, id 7, 0.10 m, 1 m away) beside them; then the same people lying on
the floor (the image turned). It checks the people, their postures, their TF frames, GetPeople,
the markers and their frame, WaitForGesture (timeout), WatchForFall (the fall) and the life cycle
of the skill nodes. Run it with the system started without a camera of its own:

    ros2 launch perception_system perception_system.launch.py camera:=none model:=color_blobs \\
      device:=cpu
    python3 check_skills.py [--report report.json]
"""

import argparse
import json
import math
import os
import sys
import threading
import time

import cv2
from geometry_msgs.msg import TransformStamped
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
import tf2_ros
from vision_msgs.msg import Detection3DArray

from perception_system.client import PerceptionClient
from perception_system_interfaces.action import WaitForGesture, WatchForFall
from perception_system_interfaces.msg import BodyEvent, PersonArray
from perception_system_interfaces.srv import GetPeople

SYSTEM = '/perception_system'
SIZE, F = 1080, 1000.0          # a square image, its focal length (pixels)
CAMERA_HEIGHT = 0.9
MARKER_ID, MARKER_SIZE, MARKER_DISTANCE = 7, 0.10, 1.0
MARKER_CENTER = (67, 100)       # pixels


def ultralytics_image(name):
    import ultralytics
    return cv2.cvtColor(cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), 'assets',
                                                name)), cv2.COLOR_BGR2RGB)


def scenes():
    """The images of the camera: standing (with the marker) and lying (on the lower part)."""
    bus = ultralytics_image('bus.jpg')                       # 810 x 1080
    standing = np.full((SIZE, SIZE, 3), 128, dtype=np.uint8)
    left = (SIZE - bus.shape[1]) // 2
    standing[:, left:left + bus.shape[1]] = bus
    side = int(round(F * MARKER_SIZE / MARKER_DISTANCE))
    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), MARKER_ID, side)
    quiet = np.full((side + 20, side + 20), 255, dtype=np.uint8)
    quiet[10:10 + side, 10:10 + side] = marker
    u, v = MARKER_CENTER
    standing[v - side // 2 - 10:v - side // 2 - 10 + side + 20,
             u - side // 2 - 10:u - side // 2 - 10 + side + 20] = quiet[..., None]
    turned = cv2.rotate(bus, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 1080 x 810
    lying = np.full((SIZE, SIZE, 3), 128, dtype=np.uint8)
    lying[SIZE - turned.shape[0]:, :] = turned
    return {'standing': standing, 'lying': lying}


def marker_in_odom():
    """Where the marker is: optical (x right, y down, z ahead) to odom (x ahead, z up)."""
    u, v = MARKER_CENTER
    x, y = (u - SIZE / 2) / F * MARKER_DISTANCE, (v - SIZE / 2) / F * MARKER_DISTANCE
    return (MARKER_DISTANCE, -x, CAMERA_HEIGHT - y)


class SkillChecker(Node):

    def __init__(self):
        super().__init__('perception_skill_checker')
        self.group = ReentrantCallbackGroup()
        self.client = PerceptionClient(self, callback_group=self.group)
        self.gesture = ActionClient(self, WaitForGesture, f'{SYSTEM}/wait_for_gesture',
                                    callback_group=self.group)
        self.fall = ActionClient(self, WatchForFall, f'{SYSTEM}/watch_for_fall',
                                 callback_group=self.group)
        self.get_people = self.create_client(GetPeople, '/body_detector/get_people',
                                             callback_group=self.group)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.static_tf = tf2_ros.StaticTransformBroadcaster(self)
        self.image_pub = self.create_publisher(Image, '/image_rgb', 10)
        self.info_pub = self.create_publisher(CameraInfo, '/camera_info', 10)
        self.scenes = scenes()
        self.scene = 'standing'
        self.lock = threading.Lock()
        self.people, self.events, self.markers = [], [], []
        self.create_subscription(PersonArray, '/people/body', self.on_people, 10,
                                 callback_group=self.group)
        self.create_subscription(BodyEvent, '/people/events', self.on_event, 50,
                                 callback_group=self.group)
        self.create_subscription(Detection3DArray, '/markers/detections', self.on_markers, 10,
                                 callback_group=self.group)
        self.results = []
        self.publish_static()
        self.create_timer(0.2, self.publish_image, callback_group=self.group)

    def publish_static(self):
        stamp = self.get_clock().now().to_msg()
        out = []
        for parent, child, xyz, q in (
                ('odom', 'camera_link', (0.0, 0.0, CAMERA_HEIGHT), (0.0, 0.0, 0.0, 1.0)),
                ('camera_link', 'camera_optical', (0.0, 0.0, 0.0), (-0.5, 0.5, -0.5, 0.5))):
            t = TransformStamped()
            t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, parent, child
            t.transform.translation.x, t.transform.translation.y, \
                t.transform.translation.z = xyz
            t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, \
                t.transform.rotation.w = q
            out.append(t)
        self.static_tf.sendTransform(out)

    def publish_image(self):
        rgb = self.scenes[self.scene]
        stamp = self.get_clock().now().to_msg()
        info = CameraInfo(width=SIZE, height=SIZE)
        info.header.stamp, info.header.frame_id = stamp, 'camera_optical'
        info.k = [F, 0.0, SIZE / 2, 0.0, F, SIZE / 2, 0.0, 0.0, 1.0]
        self.info_pub.publish(info)
        image = Image(height=SIZE, width=SIZE, encoding='rgb8', step=SIZE * 3,
                      data=rgb.tobytes())
        image.header = info.header
        self.image_pub.publish(image)

    def on_people(self, msg):
        with self.lock:
            self.people.append((self.scene, msg))

    def on_event(self, msg):
        with self.lock:
            self.events.append(msg.event)

    def on_markers(self, msg):
        with self.lock:
            self.markers.append(msg)

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

    def frame(self, name, timeout=2.0):
        try:
            t = self.tf_buffer.lookup_transform('odom', name, Time(),
                                                timeout=rclpy.duration.Duration(seconds=timeout))
        except tf2_ros.TransformException:
            return None
        p = t.transform.translation
        return p.x, p.y, p.z

    def latest(self, scene):
        with self.lock:
            found = [m for s, m in self.people if s == scene and m.people]
        return found[-1] if found else None

    def run(self):
        deadline = time.monotonic() + 90.0
        while time.monotonic() < deadline:
            try:
                if self.client.status(timeout_s=2.0).state == 'active' and \
                        self.fall.wait_for_server(timeout_sec=1.0):
                    break
            except Exception:  # noqa: B902 - not up yet
                pass
            time.sleep(0.5)
        self.check('the orchestrator and /perception_system are up',
                   self.fall.server_is_ready())
        time.sleep(2.0)

        # The skill body, on request: the people standing
        reply = self.client.start_skill('check_body', 'body', timeout_s=120.0)
        self.check('skill body: its node activated on request', reply.success and
                   '/people/body' in reply.topics, reply.message)
        time.sleep(8.0)
        msg = self.latest('standing')
        people = msg.people if msg else []
        confident = [p for p in people if p.score > 0.8]
        self.check('body: the people of the image, with their 17 keypoints',
                   len(confident) >= 3 and all(len(p.keypoints) == 17 for p in people),
                   f'{len(people)} people')
        postures = sorted(p.posture for p in confident)
        self.check('body: standing', postures.count('standing') >= 3, str(postures))
        placed = [p for p in confident if p.has_position and p.has_head and p.frame_id]
        frames = {p.frame_id: (self.frame(p.frame_id), self.frame(p.frame_id + '_head'))
                  for p in placed}
        heads_up = [f for f, (torso, head) in frames.items() if torso and head and
                    head[2] > torso[2] + 0.2 and 1.0 < torso[0] < 8.0]
        self.check('body: a TF frame per person (torso) and their head above it',
                   len(placed) >= 3 and len(heads_up) == len(placed),
                   ', '.join(f'{f} {t[0]:.1f},{t[1]:.1f},{t[2]:.1f}'
                             for f, (t, _) in frames.items() if t))
        colors = [p.clothing_color for p in confident if p.clothing_color]
        self.check('body: the color of their clothes', len(colors) >= 2, str(colors))
        self.get_people.wait_for_service(timeout_sec=5.0)
        served = self.wait(self.get_people.call_async(GetPeople.Request()), 5.0)
        self.check('body: GetPeople', served is not None and served.available and
                   len(served.people.people) >= 3)

        # WaitForGesture: nobody makes a gesture in this image
        handle = self.wait(self.gesture.send_goal_async(WaitForGesture.Goal(
            gesture='waving', timeout_s=3.0)))
        result = self.wait(handle.get_result_async(), 20.0) if handle else None
        self.check('WaitForGesture: timeout (nobody waves)', result is not None and
                   result.result.reason == 'timeout',
                   result.result.reason if result else 'no result')

        # The skill markers: the marker beside the people, and its frame
        reply = self.client.start_skill('check_markers', 'markers', timeout_s=30.0)
        time.sleep(3.0)
        with self.lock:
            seen = [d for m in self.markers[-5:] for d in m.detections]
        ids = sorted({d.results[0].hypothesis.class_id for d in seen if d.results})
        self.check('skill markers: the marker by its id', reply.success and ids == ['7'],
                   f'{reply.message}; ids {ids}')
        position = self.frame(f'marker_{MARKER_ID}')
        error = math.dist(position, marker_in_odom()) if position else None
        self.check('markers: the frame marker_7 where it is', error is not None and error < 0.02,
                   f'error {error:.3f} m' if error is not None else 'no frame')
        self.client.stop('check_markers')

        # WatchForFall while the people fall: the image turns, they lie on the floor
        handle = self.wait(self.fall.send_goal_async(WatchForFall.Goal(timeout_s=40.0)))
        self.check('WatchForFall accepted', handle is not None and handle.accepted)
        time.sleep(2.0)
        self.scene = 'lying'
        result = self.wait(handle.get_result_async(), 45.0) if handle else None
        person = result.result.person if result else None
        self.check('WatchForFall: a person fallen, with their position on the floor',
                   result is not None and result.result.fallen and person.fallen and
                   person.has_position and person.position.z < 0.6 and person.frame_id != '',
                   f'{person.id} at z {person.position.z:.2f}, {person.frame_id}'
                   if person and person.has_position else
                   (result.result.reason if result else 'no result'))
        with self.lock:
            events = list(self.events)
        self.check('body: the events of the fall (lying, fall)',
                   'lying' in events and 'fall' in events, str(sorted(set(events))))

        # The skill node goes down after the last request
        self.client.stop('check_body')
        time.sleep(2.0)
        skills = list(self.client.status().active_skills)
        self.check('skills: their nodes deactivated after the last request', not skills,
                   str(skills))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--report', default='')
    args, _ = parser.parse_known_args()
    rclpy.init()
    node = SkillChecker()
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
