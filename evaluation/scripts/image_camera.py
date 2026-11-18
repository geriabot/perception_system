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

"""A camera that publishes one image file again and again on /image_rgb (rgb8), to try a real
detection model without a robot or a simulator:

    python3 image_camera.py photo.jpg [--rate 1]
"""

import argparse

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('image')
    parser.add_argument('--rate', type=float, default=1.0)
    parser.add_argument('--topic', default='/image_rgb')
    args, _ = parser.parse_known_args()
    rgb = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    rclpy.init()
    node = Node('image_camera')
    publisher = node.create_publisher(Image, args.topic, 10)

    def publish():
        msg = Image(height=rgb.shape[0], width=rgb.shape[1], encoding='rgb8',
                    step=rgb.shape[1] * 3, data=rgb.tobytes())
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_optical'
        publisher.publish(msg)
    node.create_timer(1.0 / args.rate, publish)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
