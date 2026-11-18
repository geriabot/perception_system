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

"""The skill "markers" of the perception system: fiducial markers (ArUco, AprilTag) with their 6D
pose. A lifecycle node that /perception_system activates with the first request of the skill.

Publishes, for every image:
  /markers/detections  vision_msgs/Detection3DArray, in the frame of the image: one detection per
                       marker, its id in results[0].hypothesis.class_id ("7"), its pose (z out of
                       the marker) in bbox.center and results[0].pose, its size in bbox.size
  /markers/image       sensor_msgs/Image rgb8: the markers outlined, with their ids and axes
                       (only while someone subscribes)
  /tf                  <tf_prefix>_<id>, the pose of each marker, child of fixed_frame_id
"""

from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
import tf2_ros
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from perception_markers.markers import (MarkerDetector, matrix_to_quaternion, parse_sizes,
                                        quaternion_multiply, rotate)

IMAGE_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                       history=HistoryPolicy.KEEP_LAST, depth=1)
# The optical frame (x right, y down, z forward) in a body frame (x forward, y left, z up)
BODY_FROM_OPTICAL = matrix_to_quaternion(np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0],
                                                   [0.0, -1.0, 0.0]]))


class MarkerDetectorNode(LifecycleNode):

    def __init__(self, **kwargs):
        super().__init__('marker_detector', **kwargs)
        self.declare_parameters('', [
            ('image_topic', '/image_rgb'),
            ('camera_info_topic', '/camera_info'),
            ('camera_intrinsics', [0.0]),
            ('camera_frame_is_optical', True),
            ('camera_frame_id', ''),
            ('fixed_frame_id', 'odom'),
            ('tf_lookup_timeout_ms', 200),
            ('dictionary', 'DICT_4X4_50'),
            ('marker_size_m', 0.10),          # the side of the black square
            ('marker_sizes', ['']),           # other sizes, "id:size_m"
            ('detections_topic', '/markers/detections'),
            ('image_out_topic', '/markers/image'),
            ('publish_tf', True),
            ('tf_prefix', 'marker'),
        ])
        self.bridge = CvBridge()
        self.detector = None
        self.intrinsics = None
        self.distortion = None
        self.handles = []
        self.detections_pub = self.image_pub = None
        self.tf_buffer = self.tf_listener = self.tf_broadcaster = None
        self.processing = False
        self.image_group = MutuallyExclusiveCallbackGroup()
        self.other_group = ReentrantCallbackGroup()

    def p(self, name):
        return self.get_parameter(name).value

    def on_configure(self, state: State):
        try:
            self.detector = MarkerDetector(self.p('dictionary'), self.p('marker_size_m'),
                                           parse_sizes(self.p('marker_sizes')))
        except Exception as exc:  # noqa: B902 - any failure fails the transition
            self.get_logger().error(f'Cannot create the marker detector: {exc}')
            return TransitionCallbackReturn.FAILURE
        self.detections_pub = self.create_lifecycle_publisher(
            Detection3DArray, self.p('detections_topic'), 10)
        self.image_pub = self.create_lifecycle_publisher(Image, self.p('image_out_topic'), 10)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State):
        intrinsics = [float(v) for v in self.p('camera_intrinsics')]
        self.intrinsics = tuple(intrinsics) if len(intrinsics) == 4 and intrinsics[0] > 0 \
            else None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.handles = [self.create_subscription(Image, self.p('image_topic'), self.on_image,
                                                 IMAGE_QOS, callback_group=self.image_group)]
        if self.intrinsics is None and self.p('camera_info_topic'):
            self.handles.append(self.create_subscription(
                CameraInfo, self.p('camera_info_topic'), self.on_camera_info, IMAGE_QOS,
                callback_group=self.other_group))
        self.get_logger().info(f'Marker detector active: {self.p("dictionary")}, '
                               f'{self.p("marker_size_m")} m, images from '
                               f'{self.p("image_topic")}')
        self.processing = True
        return super().on_activate(state)

    def on_deactivate(self, state: State):
        # An image being processed now finishes with the buffer and publishers as they are; the
        # next ones are ignored
        self.processing = False
        for handle in self.handles:
            self.destroy_subscription(handle)
        self.handles = []
        self.tf_listener = None
        return super().on_deactivate(state)

    def on_cleanup(self, state: State):
        self.detector = None
        for publisher in (self.detections_pub, self.image_pub):
            if publisher is not None:
                self.destroy_lifecycle_publisher(publisher)
        self.detections_pub = self.image_pub = None
        return TransitionCallbackReturn.SUCCESS

    def on_camera_info(self, msg):
        if self.intrinsics is None and msg.k[0] > 0.0:
            self.intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])
            if len(msg.d) >= 4 and any(abs(v) > 0.0 for v in msg.d):
                self.distortion = list(msg.d)
            self.get_logger().info('Camera calibration (CameraInfo): fx={:.1f} fy={:.1f} '
                                   'cx={:.1f} cy={:.1f}'.format(*self.intrinsics))

    def camera_pose(self, frame_id, stamp):
        try:
            time = Time.from_msg(stamp)
            timeout = Duration(nanoseconds=int(self.p('tf_lookup_timeout_ms')) * 1_000_000)
            fixed = self.p('fixed_frame_id')
            if not self.tf_buffer.can_transform(fixed, frame_id, time, timeout):
                raise tf2_ros.TransformException('not available')
            return self.tf_buffer.lookup_transform(fixed, frame_id, time).transform
        except tf2_ros.TransformException as exc:
            self.get_logger().warning(f'No TF {self.p("fixed_frame_id")} -> {frame_id} at the '
                                      f'time of the image [{exc}]', throttle_duration_sec=5.0)
            return None

    def on_image(self, msg):
        if not self.processing or self.detector is None:
            return
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as exc:  # noqa: B902
            self.get_logger().error(f'cv_bridge error: {exc}', throttle_duration_sec=5.0)
            return
        markers = self.detector.detect(image, self.intrinsics, self.distortion)
        out = Detection3DArray(header=msg.header)
        optical = self.p('camera_frame_is_optical')
        posed = []
        for marker in markers:
            if marker.position is None:
                continue
            position, orientation = marker.position, marker.orientation
            if not optical:
                x, y, z = position
                position = (z, -x, -y)
                orientation = quaternion_multiply(BODY_FROM_OPTICAL, orientation)
            detection = Detection3D(header=msg.header, id=str(marker.id))
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(marker.id)
            hypothesis.hypothesis.score = 1.0
            pose = hypothesis.pose.pose
            pose.position.x, pose.position.y, pose.position.z = position
            (pose.orientation.x, pose.orientation.y, pose.orientation.z,
             pose.orientation.w) = orientation
            detection.results.append(hypothesis)
            detection.bbox.center = pose
            detection.bbox.size.x = detection.bbox.size.y = marker.size_m
            detection.bbox.size.z = 0.0
            out.detections.append(detection)
            posed.append((marker.id, position, orientation))
        self.detections_pub.publish(out)
        if posed and self.p('publish_tf'):
            self.publish_tf(msg, posed)
        if self.image_pub.get_subscription_count() > 0:
            self.image_pub.publish(self.marked_image(msg, image, markers))

    def publish_tf(self, msg, posed):
        camera_frame = self.p('camera_frame_id') or msg.header.frame_id
        transform = self.camera_pose(camera_frame, msg.header.stamp)
        if transform is None:
            return
        r = transform.rotation
        rotation = (r.x, r.y, r.z, r.w)
        transforms_out = []
        for marker_id, position, orientation in posed:
            t = TransformStamped()
            t.header.stamp = msg.header.stamp
            t.header.frame_id = self.p('fixed_frame_id')
            t.child_frame_id = f'{self.p("tf_prefix")}_{marker_id}'
            x, y, z = rotate(rotation, position)
            t.transform.translation.x = transform.translation.x + x
            t.transform.translation.y = transform.translation.y + y
            t.transform.translation.z = transform.translation.z + z
            (t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z,
             t.transform.rotation.w) = quaternion_multiply(rotation, orientation)
            transforms_out.append(t)
        self.tf_broadcaster.sendTransform(transforms_out)

    def marked_image(self, msg, image, markers):
        import cv2
        out = image.copy()
        if markers:
            cv2.aruco.drawDetectedMarkers(
                out, [m.corners.reshape(1, 4, 2).astype(np.float32) for m in markers],
                np.array([[m.id] for m in markers]))
        for marker in markers:
            if marker.position is not None:
                x, y = marker.corners.mean(axis=0)
                cv2.putText(out, f'{marker.distance_m:.2f} m', (int(x), int(y) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 2, cv2.LINE_AA)
        image_msg = self.bridge.cv2_to_imgmsg(out, 'rgb8')
        image_msg.header = msg.header
        return image_msg


def main(args=None):
    rclpy.init(args=args)
    node = MarkerDetectorNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:  # noqa: B902 - re-raised unless the context has ended
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
