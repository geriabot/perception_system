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

"""A synthetic RGB-D camera on a robot that turns, looking at colored upright boxes ("people") at
known places: the ground truth of the checks of the perception system, with no simulator and no
GPU. The color_blobs model detects the boxes by their color.

Publishes (rate images per second, all with the same stamp):
  /image_rgb                 sensor_msgs/Image rgb8, 640x480
  /rgbd_camera/depth_image   sensor_msgs/Image 32FC1 (m along the optical axis; wall at 6 m)
  /rgbd_camera/camera_info   sensor_msgs/CameraInfo
  /tf                        odom -> base_link (the robot turns left and right), at 50 Hz
  /tf_static                 base_link -> camera_link -> camera_optical
  /synthetic/ground_truth    std_msgs/String, JSON: stamp, yaw, and each box in odom

The scene (odom, m): a red box walking left and right 2.5 m ahead (y from -1 to 0), a red box
still at (4, 0.8), and a blue box at (3.5, -1.2), which the walker sometimes hides in part. Each
box is 0.5 m wide and 1.7 m tall, standing on the floor: the center of its image is 0.85 m high.
The robot turns its base by yaw_amplitude * sin(0.25 t).

A box leaves the scene while its name is in the last message of /synthetic/hidden (std_msgs/String,
JSON list of names, e.g. ["walker"]), so that a check can make an object leave and come back.
"""

import argparse
import json
import math

from geometry_msgs.msg import TransformStamped
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

WIDTH, HEIGHT, F = 640, 480, 525.0
CX, CY = 319.5, 239.5


def set_resolution(scale):
    """The camera scaled (0.5: 320x240, the same field of view)."""
    global WIDTH, HEIGHT, F, CX, CY
    WIDTH, HEIGHT, F = int(640 * scale), int(480 * scale), 525.0 * scale
    CX, CY = (WIDTH - 1) / 2.0, (HEIGHT - 1) / 2.0
CAMERA_HEIGHT = 1.0          # m above the floor
CAMERA_FORWARD = 0.1         # m ahead of base_link
BOX_WIDTH, BOX_HEIGHT = 0.5, 1.7
WALL = 6.0
COLORS = {'red': (220, 20, 20), 'blue': (20, 20, 220)}


def boxes_at(t, hidden=()):
    """The boxes in odom at time t (s since the start): (name, color, x, y)."""
    return [box for box in [
        ('walker', 'red', 2.5, -0.5 + 0.5 * math.sin(0.4 * t)),
        ('still', 'red', 4.0, 0.8),
        ('blue', 'blue', 3.5, -1.2),
    ] if box[0] not in hidden]


def quaternion_yaw(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def to_camera(x, y, yaw):
    """A point of the floor plane of odom (x, y) in the frame of the camera: forward, left."""
    cx, cy = CAMERA_FORWARD * math.cos(yaw), CAMERA_FORWARD * math.sin(yaw)
    dx, dy = x - cx, y - cy
    forward = math.cos(yaw) * dx + math.sin(yaw) * dy
    left = -math.sin(yaw) * dx + math.cos(yaw) * dy
    return forward, left


def render(t, yaw, hidden=()):
    rgb = np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8)
    depth = np.full((HEIGHT, WIDTH), WALL, dtype=np.float32)
    placed = []
    for name, color, x, y in boxes_at(t, hidden):
        forward, left = to_camera(x, y, yaw)
        if forward <= 0.3:
            continue
        u = CX - F * left / forward
        half = F * (BOX_WIDTH / 2.0) / forward
        top = CY - F * (BOX_HEIGHT - CAMERA_HEIGHT) / forward
        bottom = CY + F * CAMERA_HEIGHT / forward
        u0, u1 = int(round(u - half)), int(round(u + half))
        v0, v1 = int(round(top)), int(round(bottom))
        placed.append((forward, name, color, u0, u1, v0, v1))
    for forward, name, color, u0, u1, v0, v1 in sorted(placed, reverse=True):   # far to near
        u0c, u1c = max(u0, 0), min(u1, WIDTH)
        v0c, v1c = max(v0, 0), min(v1, HEIGHT)
        if u1c <= u0c or v1c <= v0c:
            continue
        rgb[v0c:v1c, u0c:u1c] = COLORS[color]
        depth[v0c:v1c, u0c:u1c] = forward
    return rgb, depth


class SyntheticCamera(Node):

    def __init__(self, rate, yaw_amplitude, duration, warmup=0.0):
        super().__init__('synthetic_camera')
        self.yaw_amplitude = yaw_amplitude
        # Images only after warmup seconds (the TF from the start): the subscribers discover the
        # publishers first, so that none misses the first images
        self.warmup = warmup
        self.duration = duration
        self.start = self.get_clock().now()
        self.rgb_pub = self.create_publisher(Image, '/image_rgb', 10)
        self.depth_pub = self.create_publisher(Image, '/rgbd_camera/depth_image', 10)
        self.info_pub = self.create_publisher(CameraInfo, '/rgbd_camera/camera_info', 10)
        self.truth_pub = self.create_publisher(
            String, '/synthetic/ground_truth',
            QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.VOLATILE))
        self.tf = TransformBroadcaster(self)
        self.static_tf = StaticTransformBroadcaster(self)
        self.publish_static()
        self.create_timer(0.02, self.publish_odom)
        self.create_timer(1.0 / rate, self.publish_images)
        self.frames = 0
        self.hidden = []
        self.create_subscription(String, '/synthetic/hidden', self.on_hidden, 10)

    def on_hidden(self, msg):
        self.hidden = list(json.loads(msg.data or '[]'))
        self.get_logger().info(f'Hidden: {self.hidden}')

    def elapsed(self, stamp):
        return (stamp - self.start).nanoseconds / 1e9

    def yaw(self, t):
        return self.yaw_amplitude * math.sin(0.25 * t)

    def transform(self, stamp, parent, child, xyz, q):
        msg = TransformStamped()
        msg.header.stamp = stamp.to_msg()
        msg.header.frame_id, msg.child_frame_id = parent, child
        msg.transform.translation.x, msg.transform.translation.y, \
            msg.transform.translation.z = xyz
        msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, \
            msg.transform.rotation.w = q
        return msg

    def publish_static(self):
        now = self.get_clock().now()
        self.static_tf.sendTransform([
            self.transform(now, 'base_link', 'camera_link', (CAMERA_FORWARD, 0.0, CAMERA_HEIGHT),
                           (0.0, 0.0, 0.0, 1.0)),
            # Optical: z forward, x right, y down
            self.transform(now, 'camera_link', 'camera_optical', (0.0, 0.0, 0.0),
                           (-0.5, 0.5, -0.5, 0.5)),
        ])

    def publish_odom(self):
        now = self.get_clock().now()
        self.tf.sendTransform(self.transform(now, 'odom', 'base_link', (0.0, 0.0, 0.0),
                                             quaternion_yaw(self.yaw(self.elapsed(now)))))

    def publish_images(self):
        now = self.get_clock().now()
        t = self.elapsed(now)
        if t < self.warmup:
            return
        if self.duration > 0 and t > self.duration:
            raise SystemExit
        yaw = self.yaw(t)
        hidden = list(self.hidden)
        rgb, depth = render(t, yaw, hidden)
        header_stamp = now.to_msg()
        info = CameraInfo(width=WIDTH, height=HEIGHT)
        info.header.stamp, info.header.frame_id = header_stamp, 'camera_optical'
        info.k = [F, 0.0, CX, 0.0, F, CY, 0.0, 0.0, 1.0]
        info.p = [F, 0.0, CX, 0.0, 0.0, F, CY, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.info_pub.publish(info)
        d = Image(height=HEIGHT, width=WIDTH, encoding='32FC1', step=WIDTH * 4)
        d.header = info.header
        d.data = depth.tobytes()
        self.depth_pub.publish(d)
        image = Image(height=HEIGHT, width=WIDTH, encoding='rgb8', step=WIDTH * 3)
        image.header = info.header
        image.data = rgb.tobytes()
        self.rgb_pub.publish(image)
        self.truth_pub.publish(String(data=json.dumps({
            'stamp': [header_stamp.sec, header_stamp.nanosec], 'frame': self.frames, 'yaw': yaw,
            'boxes': [{'name': n, 'color': c, 'x': x, 'y': y} for n, c, x, y in boxes_at(t, hidden)]})))
        self.frames += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rate', type=float, default=5.0, help='images per second')
    parser.add_argument('--yaw-amplitude', type=float, default=0.15,
                        help='rad: the robot turns by yaw_amplitude * sin(0.25 t)')
    parser.add_argument('--duration', type=float, default=0.0, help='s (0: forever)')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='image size: 1.0 is 640x480, 0.5 is 320x240')
    parser.add_argument('--warmup', type=float, default=0.0,
                        help='s without images at the start (the TF is published)')
    args, _ = parser.parse_known_args()
    set_resolution(args.scale)
    rclpy.init()
    node = SyntheticCamera(args.rate, args.yaw_amplitude, args.duration, args.warmup)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
