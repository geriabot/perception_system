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

"""The skill "body" of the perception system: the people in view with the keypoints of their
body, their posture, a fall, their gestures, where they point, whether they face the camera and
the color of their clothes. A lifecycle node that /perception_system activates with the first
request of the skill and deactivates after the last one (it runs no model while nobody needs it).

Publishes, for every image:
  /people/body         perception_system_interfaces/PersonArray: every person
  /people/events       perception_system_interfaces/BodyEvent: a gesture, posture, fall started
  /people/body_image   sensor_msgs/Image rgb8: the skeletons, ids, postures and gestures (only
                       while someone subscribes)
  /tf                  <tf_prefix>_<id> (the torso), <tf_prefix>_<id>_head and, while pointing at
                       the floor, <tf_prefix>_<id>_pointed, children of fixed_frame_id
Serves ~/get_people (GetPeople): the people of the last image.

The 3D placement: with a depth image aligned with the color image (depth_image_topic), each
keypoint at its depth; without it, the whole body at the distance that the size of its torso
gives. The camera pose at the time of the image places them in the fixed frame (z up, the floor at
body.floor_z).
"""

import threading

from cv_bridge import CvBridge
from geometry_msgs.msg import Point, TransformStamped, Vector3
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

from perception_body import body as b
from perception_body.drawing import draw_people
from perception_system_interfaces.msg import BodyEvent, BodyKeypoint, Person, PersonArray
from perception_system_interfaces.srv import GetPeople

IMAGE_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                       history=HistoryPolicy.KEEP_LAST, depth=1)

BODY_DEFAULTS = {name: value for name, value in vars(b.BodyParameters()).items()}


class BodyDetector(LifecycleNode):

    def __init__(self, **kwargs):
        super().__init__('body_detector', **kwargs)
        self.declare_parameters('', [
            ('image_topic', '/image_rgb'),
            # The 3D placement: the calibration (or fixed intrinsics [fx, fy, cx, cy]) and,
            # optionally, a depth image aligned with the color image
            ('camera_info_topic', '/camera_info'),
            ('camera_intrinsics', [0.0]),
            ('depth_image_topic', ''),
            ('camera_frame_is_optical', True),
            ('camera_frame_id', ''),          # instead of the frame of the images
            ('fixed_frame_id', 'odom'),
            ('tf_lookup_timeout_ms', 200),
            # The pose model (pose.py)
            ('model_id', ''),
            ('device', ''),
            ('imgsz', 640),
            ('threshold', 0.4),
            ('max_age', 30),
            # Outputs
            ('people_topic', '/people/body'),
            ('events_topic', '/people/events'),
            ('image_out_topic', '/people/body_image'),
            ('publish_tf', True),
            ('tf_prefix', 'body'),
        ])
        self.declare_parameters('body', list(BODY_DEFAULTS.items()))
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.detector = None
        self.analyzer = None
        self.intrinsics = None
        self.last_depth = None
        self.last_people = None
        self.handles = []
        self.people_pub = self.events_pub = self.image_pub = None
        self.tf_buffer = self.tf_listener = self.tf_broadcaster = None
        self.processing = False
        self.image_group = MutuallyExclusiveCallbackGroup()
        self.other_group = ReentrantCallbackGroup()

    def p(self, name):
        return self.get_parameter(name).value

    # ------------------------------------------------------------------ lifecycle
    def on_configure(self, state: State):
        try:
            from perception_body.pose import PoseDetector
            self.detector = PoseDetector(
                self.p('model_id'), self.p('device'), self.p('imgsz'), self.p('threshold'),
                {'max_age': self.p('max_age')}, self.get_logger().info)
        except Exception as exc:  # noqa: B902 - any failure fails the transition
            self.get_logger().error(f'Cannot load the pose model: {exc}')
            return TransitionCallbackReturn.FAILURE
        self.people_pub = self.create_lifecycle_publisher(PersonArray, self.p('people_topic'),
                                                          10)
        self.events_pub = self.create_lifecycle_publisher(BodyEvent, self.p('events_topic'), 10)
        self.image_pub = self.create_lifecycle_publisher(Image, self.p('image_out_topic'), 10)
        intrinsics = [float(v) for v in self.p('camera_intrinsics')]
        self.fixed_intrinsics = tuple(intrinsics) if len(intrinsics) == 4 and \
            intrinsics[0] > 0.0 else None
        self.get_logger().info(f'Pose model loaded ({self.detector.device})')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State):
        params = b.BodyParameters(**{name: type(default)(self.p(f'body.{name}'))
                                     for name, default in BODY_DEFAULTS.items()})
        self.analyzer = b.BodyAnalyzer(params)
        self.intrinsics = self.fixed_intrinsics
        self.last_depth = None
        self.last_people = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.handles = [
            self.create_subscription(Image, self.p('image_topic'), self.on_image, IMAGE_QOS,
                                     callback_group=self.image_group),
            self.create_service(GetPeople, '~/get_people', self.on_get_people,
                                callback_group=self.other_group),
        ]
        if self.intrinsics is None and self.p('camera_info_topic'):
            self.handles.append(self.create_subscription(
                CameraInfo, self.p('camera_info_topic'), self.on_camera_info, IMAGE_QOS,
                callback_group=self.other_group))
        if self.p('depth_image_topic'):
            self.handles.append(self.create_subscription(
                Image, self.p('depth_image_topic'), self.on_depth, IMAGE_QOS,
                callback_group=self.other_group))
        self.get_logger().info(
            f'Body detector active: images from {self.p("image_topic")}, 3D '
            f'{"with " + self.p("depth_image_topic") if self.p("depth_image_topic") else "from the size of the body"}')
        self.processing = True
        return super().on_activate(state)

    def on_deactivate(self, state: State):
        # An image being processed now finishes with the buffer and publishers as they are; the
        # next ones are ignored
        self.processing = False
        for handle in self.handles:
            if hasattr(handle, 'topic_name'):
                self.destroy_subscription(handle)
            else:
                self.destroy_service(handle)
        self.handles = []
        self.tf_listener = None
        return super().on_deactivate(state)

    def on_cleanup(self, state: State):
        self.detector = None
        for publisher in (self.people_pub, self.events_pub, self.image_pub):
            if publisher is not None:
                self.destroy_lifecycle_publisher(publisher)
        self.people_pub = self.events_pub = self.image_pub = None
        return TransitionCallbackReturn.SUCCESS

    # ------------------------------------------------------------------ inputs
    def on_camera_info(self, msg):
        if self.intrinsics is None and msg.k[0] > 0.0:
            self.intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])
            self.get_logger().info('Camera calibration (CameraInfo): fx={:.1f} fy={:.1f} '
                                   'cx={:.1f} cy={:.1f}'.format(*self.intrinsics))

    def on_depth(self, msg):
        self.last_depth = msg

    def on_get_people(self, request, response):
        with self.lock:
            response.available = self.last_people is not None
            if self.last_people is not None:
                response.people = self.last_people
        return response

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
                                      f'time of the image [{exc}]: no 3D',
                                      throttle_duration_sec=5.0)
            return None

    # ------------------------------------------------------------------ an image
    def on_image(self, msg):
        if not self.processing or self.analyzer is None or self.detector is None:
            return
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as exc:  # noqa: B902
            self.get_logger().error(f'cv_bridge error: {exc}', throttle_duration_sec=5.0)
            return
        detections = [d for d in self.detector.detect(image) if d.id]
        camera_frame = self.p('camera_frame_id') or msg.header.frame_id
        transform = self.camera_pose(camera_frame, msg.header.stamp) \
            if self.intrinsics is not None and detections else None
        depth = None
        if transform is not None and self.last_depth is not None:
            from detection_3d_estimators.node import depth_to_metres
            try:
                depth = depth_to_metres(self.last_depth)
                if depth.shape != image.shape[:2]:
                    depth = None
            except ValueError as exc:
                self.get_logger().warning(str(exc), throttle_duration_sec=10.0)
        p = self.analyzer.p
        observations = []
        for d in detections:
            o = b.Observation(d.id, d.keypoints, d.box,
                              clothing_color=b.clothing_color(image, d.keypoints, p))
            if transform is not None:
                points, valid, from_depth = b.place_keypoints(
                    d.keypoints, self.intrinsics, p.keypoint_threshold, depth)
                if valid.any():
                    if not self.p('camera_frame_is_optical'):
                        points = np.stack([points[:, 2], -points[:, 0], -points[:, 1]], 1)
                    t, r = transform.translation, transform.rotation
                    o.points = b.transform_points((t.x, t.y, t.z), (r.x, r.y, r.z, r.w),
                                                  points)
                    o.valid, o.from_depth = valid, from_depth
            observations.append(o)
        t = Time.from_msg(msg.header.stamp).nanoseconds / 1e9
        reports, events = self.analyzer.update(t, observations)
        people = PersonArray(header=msg.header)
        people.fixed_frame_id = self.p('fixed_frame_id') if transform is not None else ''
        transforms_out = []
        for d, o, report in zip(detections, observations, reports):
            person = self.person_message(d, o, report)
            if transform is not None and self.p('publish_tf'):
                transforms_out += self.person_transforms(msg.header.stamp, person)
            people.people.append(person)
        with self.lock:
            self.last_people = people
        self.people_pub.publish(people)
        by_id = {person.id: person for person in people.people}
        for event, person_id in events:
            self.events_pub.publish(BodyEvent(event=event, header=msg.header,
                                              fixed_frame_id=people.fixed_frame_id,
                                              person=by_id[person_id]))
            self.get_logger().info(f'Person {person_id}: {event}')
        if transforms_out:
            self.tf_broadcaster.sendTransform(transforms_out)
        if self.image_pub.get_subscription_count() > 0:
            out = self.bridge.cv2_to_imgmsg(draw_people(image, detections, reports), 'rgb8')
            out.header = msg.header
            self.image_pub.publish(out)

    def person_message(self, d, o, report):
        person = Person(id=d.id, score=float(d.score), posture=report.posture,
                        fallen=report.fallen, gestures=list(report.gestures),
                        facing_camera=bool(report.facing_camera),
                        clothing_color=o.clothing_color)
        x0, y0, x1, y1 = d.box
        person.bbox.center.position.x = (x0 + x1) / 2.0
        person.bbox.center.position.y = (y0 + y1) / 2.0
        person.bbox.size_x, person.bbox.size_y = x1 - x0, y1 - y0
        threshold = self.analyzer.p.keypoint_threshold
        for i, name in enumerate(b.KEYPOINT_NAMES):
            x, y, score = (float(v) for v in d.keypoints[i])
            keypoint = BodyKeypoint(name=name, x=x, y=y, score=score)
            if o.points is not None and o.valid[i] and score >= threshold:
                keypoint.has_position = True
                keypoint.position = Point(x=float(o.points[i][0]), y=float(o.points[i][1]),
                                          z=float(o.points[i][2]))
            person.keypoints.append(keypoint)
        for attribute, flag in (('position', 'has_position'), ('head', 'has_head'),
                                ('pointed_location', 'has_pointed_location')):
            value = getattr(report, attribute)
            if value is not None:
                setattr(person, flag, True)
                setattr(person, attribute, Point(x=float(value[0]), y=float(value[1]),
                                                 z=float(value[2])))
        if report.pointing_direction is not None:
            person.pointing = True
            direction = report.pointing_direction
            person.pointing_direction = Vector3(x=float(direction[0]), y=float(direction[1]),
                                                z=float(direction[2]))
        if person.has_position and self.p('publish_tf'):
            person.frame_id = f'{self.p("tf_prefix")}_{d.id}'
        return person

    def person_transforms(self, stamp, person):
        out = []
        if not person.frame_id:
            return out
        for suffix, flag, point in (('', 'has_position', person.position),
                                    ('_head', 'has_head', person.head),
                                    ('_pointed', 'has_pointed_location',
                                     person.pointed_location)):
            if not getattr(person, flag):
                continue
            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = self.p('fixed_frame_id')
            transform.child_frame_id = person.frame_id + suffix
            transform.transform.translation.x = point.x
            transform.transform.translation.y = point.y
            transform.transform.translation.z = point.z
            transform.transform.rotation.w = 1.0
            out.append(transform)
        return out


def main(args=None):
    rclpy.init(args=args)
    node = BodyDetector()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:  # noqa: B902 - re-raised unless the context has ended
        # A callback that was running when the process was asked to stop publishes after the
        # context has ended: not an error. Raised if it is not that
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
