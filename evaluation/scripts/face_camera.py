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

"""A synthetic camera that shows a drawn face, silent or talking: the images for the skills about
the user (the lip activity) without a camera and without a person. MediaPipe's face landmarker
finds the drawn face, and its lip gap follows the drawn mouth.

Publishes /image_rgb (sensor_msgs/Image rgb8, 640x480) at --rate images per second. The mouth is
closed while the face is silent; while it talks, it opens and closes several times per second
with an irregular amplitude, as the lips do in speech. The face talks while the last message on
/face_camera/talking (std_msgs/Bool) is true (--talking: from the start); --no-face shows the
background only.

    python3 face_camera.py [--rate 15] [--talking] [--no-face]
"""

import argparse
import math
import random

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

WIDTH, HEIGHT = 640, 480


def draw_face(open_px, face=True):
    """The image (RGB) of a face whose mouth is open open_px pixels."""
    img = np.full((HEIGHT, WIDTH, 3), (220, 210, 200), np.uint8)
    if not face:
        return img
    cx, cy = WIDTH // 2, HEIGHT // 2
    cv2.ellipse(img, (cx, cy - 20), (110, 145), 0, 0, 360, (225, 175, 150), -1)   # skin
    cv2.ellipse(img, (cx, cy - 165), (115, 60), 0, 180, 360, (70, 50, 40), -1)    # hair
    for dx in (-45, 45):
        cv2.ellipse(img, (cx + dx, cy - 50), (24, 12), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(img, (cx + dx, cy - 50), 9, (30, 40, 60), -1)
        cv2.line(img, (cx + dx - 28, cy - 80), (cx + dx + 28, cy - 84), (70, 50, 40), 6)
    nose = np.array([[cx, cy - 40], [cx - 16, cy + 18], [cx + 16, cy + 18]])
    cv2.polylines(img, [nose], False, (180, 130, 110), 3)
    cv2.ellipse(img, (cx, cy + 62), (42, 6 + open_px), 0, 0, 360, (140, 40, 60), -1)  # lips
    if open_px > 2:
        cv2.ellipse(img, (cx, cy + 62), (34, max(1, open_px - 2)), 0, 0, 360, (40, 20, 30), -1)
    return img


class FaceCamera(Node):

    def __init__(self, rate, talking, face):
        super().__init__('face_camera')
        self.talking = talking
        self.face = face
        self.random = random.Random(0)
        self.phase = 0.0
        self.amplitude = 14.0
        self.period = 1.0 / rate
        self.publisher = self.create_publisher(Image, '/image_rgb', 10)
        self.create_subscription(Bool, '/face_camera/talking', self.on_talking, 10)
        self.create_timer(self.period, self.publish)
        self.get_logger().info(f'A drawn face on /image_rgb at {rate} images/s, '
                               f'{"talking" if talking else "silent"}')

    def on_talking(self, msg):
        if msg.data != self.talking:
            self.get_logger().info('talking' if msg.data else 'silent')
        self.talking = msg.data

    def mouth(self):
        if not self.talking:
            return 0
        # Syllables of about 4 per second, each with its own amplitude
        previous = self.phase
        self.phase += 2 * math.pi * 4.0 * self.period
        if math.floor(previous / math.pi) != math.floor(self.phase / math.pi):
            self.amplitude = self.random.uniform(8.0, 20.0)
        return int(round(self.amplitude * abs(math.sin(self.phase))))

    def publish(self):
        img = draw_face(self.mouth(), self.face)
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_optical'
        msg.height, msg.width = HEIGHT, WIDTH
        msg.encoding = 'rgb8'
        msg.step = WIDTH * 3
        msg.data = img.tobytes()
        self.publisher.publish(msg)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--rate', type=float, default=15.0)
    parser.add_argument('--talking', action='store_true')
    parser.add_argument('--no-face', action='store_true')
    args = parser.parse_args()
    rclpy.init()
    node = FaceCamera(args.rate, args.talking, not args.no_face)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
