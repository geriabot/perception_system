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

"""/perception_system: the perception capability of the robot, a lifecycle node.

Its clients (the behaviors of the robot, the attention system, the interaction system...) open
requests and close them; each request has its own outputs, so they do not interfere:

  ~/start_detection     StartDetection: detect and track a class in every image; its topics
  ~/start_target_frame  StartTargetFrame: a TF frame at a tracked detection or a group
  ~/start_skill         StartSkill: keep a skill node active (the user speaking, ...)
  ~/stop                StopRequest: end a request by its name
  ~/detect_once         DetectOnce: the detections of a class in the next image
  ~/get_objects         GetObjects: what a detection request saw in its last image
  ~/list_models         ListModels: the detection models and their classes
  ~/get_user_state      GetUserState: the latest state of the user
  ~/get_status, ~/status   PerceptionStatus: the open requests and the detector's speed

The request services exist while the node is active; deactivating it ends every request. The
parameters are described in docs/CONFIGURATION.md.
"""

import statistics
import threading
import time

from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, TransformStamped
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float32, Float64, Header, String
import tf2_ros
from vision_msgs.msg import Detection2D, Detection2DArray, Detection3DArray, \
    ObjectHypothesisWithPose
from visualization_msgs.msg import MarkerArray

from perception_detectors import marking
from perception_detectors.tracking import TRACKER_DEFAULTS
from perception_system import projection, requests as rq, targets, transforms
from perception_system.engine import LocalEngine, RemoteEngine, RequestConfig
from perception_system.skills import SkillManager
from perception_system_interfaces.msg import PerceptionStatus, RequestStatus, TrackedObject, \
    UserState
from perception_system_interfaces.srv import DetectOnce, GetObjects, GetStatus, GetUserState, \
    ListModels, StartDetection, StartSkill, StartTargetFrame, StopRequest

IMAGE_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                       durability=DurabilityPolicy.VOLATILE)
LATCHED_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
TRACKER_PARAMETERS = {k: v for k, v in TRACKER_DEFAULTS.items()
                      if k not in ('tracker_type', 'device')}
REMOTE_DEFAULTS = {
    'server_url': 'http://localhost:8765',
    'token_env': '',
    'image_format': 'jpeg',
    'jpeg_quality': 90,
    'timeout_s': 2.0,
    'wait_for_server_s': 60.0,
    'connect_timeout_s': 600.0,
    'added_latency_ms': 0.0,
}


class PerceptionSystem(LifecycleNode):

    def __init__(self, **kwargs):
        super().__init__('perception_system', **kwargs)
        self.declare_parameters('', [
            ('image_topic', '/image_rgb'),
            ('engine', 'local'),                   # local | remote
            ('model', 'omdet'),
            ('model_options', ''),                 # YAML
            ('device', ''),                        # '' = cuda if available
            ('tracker', 'strongsort'),             # strongsort | bytetrack | none
            ('preload_model', True),
            ('zoom_small_detections', False),
            ('default_namespace', '/perception_system/requests'),
            ('frames_namespace', '/perception_system/frames'),
            ('fixed_frame_id', 'odom'),
            ('tf_lookup_timeout_ms', 1000),
            ('frame_period_ms', 100),
            ('skill_names', ['speaking', 'emotion']),
            ('skill_nodes', ['lip_activity_detector', 'emotion_detector']),
            ('skill_topics', ['/user_speaking,/lip_activity', '/user_emotion']),
            ('autostart_skills', ['']),
            ('skill_timeout_s', 10.0),
            ('user_speaking_topic', '/user_speaking'),
            ('user_emotion_topic', '/user_emotion'),
            ('status_period_s', 1.0),
            ('stats_period_s', 30.0),
            ('autostart', False),     # configure and activate itself at start
            # Detection requests opened on activation, without a class (images only until a
            # client sets one): "name" or "name:output_namespace" ("attention:/" at the root)
            ('startup_requests', ['']),
        ])
        self.declare_parameters('tracking', list(TRACKER_PARAMETERS.items()))
        self.declare_parameters('remote', list(REMOTE_DEFAULTS.items()))
        self.declare_parameters('projection', list(projection.DEFAULTS.items()))

        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.requests = {}           # name -> DetectionRequest | FrameRequest | SkillRequest
        self.once_waiters = {}       # internal request name -> [event, result]
        self.once_count = 0
        self.engine = None
        self.projector = None
        self.skills = None
        self.active = False
        self.frame_times, self.failures, self.stats_start = [], 0, time.monotonic()
        self.stats = (0.0, 0.0, 0)
        self.lifecycle_state = 'unconfigured'

        self.image_group = MutuallyExclusiveCallbackGroup()
        self.service_group = ReentrantCallbackGroup()
        self.sensor_group = ReentrantCallbackGroup()
        self.client_group = ReentrantCallbackGroup()
        self.status_group = MutuallyExclusiveCallbackGroup()

        # Always available: the state of the system and of the user
        self.status_pub = self.create_publisher(PerceptionStatus, '~/status', LATCHED_QOS)
        self.create_service(GetStatus, '~/get_status', self.on_get_status,
                            callback_group=self.service_group)
        self.user_state = UserState()
        self.user_state_available = False
        self.create_subscription(String, self.get_parameter('user_emotion_topic').value,
                                 self.on_emotion, 10, callback_group=self.sensor_group)
        self.create_subscription(Bool, self.get_parameter('user_speaking_topic').value,
                                 self.on_speaking, 10, callback_group=self.sensor_group)
        self.create_service(GetUserState, '~/get_user_state', self.on_get_user_state,
                            callback_group=self.service_group)
        self.status_timer = self.create_timer(
            self.get_parameter('status_period_s').value, self.publish_status,
            callback_group=self.status_group)

        self.handles = []            # subscriptions and services of the active state
        self.frame_time_pub = None
        self.tf_buffer = None
        self.tf_listener = None
        self.tf_broadcaster = None

    # ------------------------------------------------------------------ parameters
    def p(self, name):
        return self.get_parameter(name).value

    def group_parameters(self, prefix, defaults):
        return {name: self.p(f'{prefix}.{name}') for name in defaults}

    def defaults(self):
        import yaml
        options = yaml.safe_load(self.p('model_options')) if self.p('model_options') else {}
        if not isinstance(options, dict):
            raise ValueError('model_options must be a YAML mapping, e.g. "{threshold: 0.35}"')
        return {'model': self.p('model'), 'model_options': options, 'tracker': self.p('tracker')}

    # ------------------------------------------------------------------ lifecycle
    def on_configure(self, state: State):
        try:
            self.defaults()
            tracker_parameters = dict(self.group_parameters('tracking', TRACKER_PARAMETERS),
                                      device=self.p('device'))
            if self.p('engine') == 'local':
                self.engine = LocalEngine(self.p('device'), tracker_parameters,
                                          log=self.get_logger().info)
            elif self.p('engine') == 'remote':
                self.engine = RemoteEngine(
                    log=self.get_logger().info, tracker_parameters=tracker_parameters,
                    **self.group_parameters('remote', REMOTE_DEFAULTS))
            else:
                raise ValueError(f'engine must be local or remote, not {self.p("engine")}')
            params = self.group_parameters('projection', projection.DEFAULTS)
            if params['estimator'] != 'none':
                self.projector = projection.Projector(params)
                self.get_logger().info(f'3D placement: {params["estimator"]}, '
                                       f'{self.projector.description}')
            else:
                self.get_logger().info('3D placement: none (projection.estimator)')
            self.skills = SkillManager(self, self.p('skill_names'), self.p('skill_nodes'),
                                       self.p('skill_topics'), self.client_group,
                                       self.p('skill_timeout_s'))
        except Exception as exc:  # noqa: B902 - any failure fails the transition
            self.get_logger().error(f'Cannot configure the perception system: {exc}')
            return TransitionCallbackReturn.FAILURE
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.set_lifecycle_state('inactive')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State):
        model = self.p('model')
        self.get_logger().info(f'Activating the perception system ({self.p("engine")} engine, '
                               f'model {model})')
        try:
            if isinstance(self.engine, RemoteEngine):
                self.engine.connect()
            if self.p('preload_model'):
                self.engine.load(model, self.defaults()['model_options'])
        except Exception as exc:  # noqa: B902 - any failure fails the activation
            self.get_logger().error(f'Cannot start the detector: {exc}')
            return TransitionCallbackReturn.ERROR

        sub = self.create_subscription
        self.handles = [sub(Image, self.p('image_topic'), self.on_image, IMAGE_QOS,
                            callback_group=self.image_group)]
        if self.projector is not None:
            if self.projector.needs_depth:
                self.handles.append(sub(
                    Image, self.p('projection.depth_image_topic'), self.projector.set_depth,
                    qos_profile_sensor_data, callback_group=self.sensor_group))
            if not self.projector.ready():
                self.camera_info_sub = sub(
                    CameraInfo, self.p('projection.camera_info_topic'), self.on_camera_info,
                    QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.VOLATILE),
                    callback_group=self.sensor_group)
                self.handles.append(self.camera_info_sub)
                self.get_logger().info('Waiting for the camera calibration on '
                                       f'{self.p("projection.camera_info_topic")}')
        srv = self.create_service
        group = self.service_group
        self.handles += [
            srv(StartDetection, '~/start_detection', self.on_start_detection,
                callback_group=group),
            srv(StartTargetFrame, '~/start_target_frame', self.on_start_target_frame,
                callback_group=group),
            srv(StartSkill, '~/start_skill', self.on_start_skill, callback_group=group),
            srv(StopRequest, '~/stop', self.on_stop, callback_group=group),
            srv(DetectOnce, '~/detect_once', self.on_detect_once, callback_group=group),
            srv(GetObjects, '~/get_objects', self.on_get_objects, callback_group=group),
            srv(ListModels, '~/list_models', self.on_list_models, callback_group=group),
        ]
        self.frame_time_pub = self.create_publisher(Float32, '~/frame_time', 10)
        self.active = True
        for entry in [e for e in self.p('startup_requests') if e]:
            name, _, namespace = entry.partition(':')
            response = self.on_start_detection(
                StartDetection.Request(name=name.strip(), output_namespace=namespace.strip()),
                StartDetection.Response())
            if not response.success:
                self.get_logger().error(f'Startup request {entry}: {response.message}')
                self.active = False
                return TransitionCallbackReturn.ERROR
        self.set_lifecycle_state('active')
        autostart = [s for s in self.p('autostart_skills') if s]
        if autostart:
            # After the transition: the skill nodes answer through this executor
            self.autostart_timer = self.create_timer(
                0.1, lambda: self.autostart_skills(autostart), callback_group=self.service_group)
        self.get_logger().info(f'Perception system active, images from {self.p("image_topic")}')
        return super().on_activate(state)

    def autostart_skills(self, names):
        self.destroy_timer(self.autostart_timer)
        for skill in names:
            response = StartSkill.Response()
            self.on_start_skill(StartSkill.Request(name=f'autostart_{skill}', skill=skill),
                                response)
            log = self.get_logger().info if response.success else self.get_logger().error
            log(response.message)

    def on_deactivate(self, state: State):
        self.active = False
        with self.lock:
            for name in list(self.requests):
                self.close_request(name)
        for handle in self.handles:
            if hasattr(handle, 'srv_name'):
                self.destroy_service(handle)
            else:
                self.destroy_subscription(handle)
        self.handles = []
        if self.frame_time_pub is not None:
            self.destroy_publisher(self.frame_time_pub)
            self.frame_time_pub = None
        self.engine.shutdown()
        self.set_lifecycle_state('inactive')
        self.get_logger().info('Perception system deactivated: every request ended')
        return super().on_deactivate(state)

    def on_cleanup(self, state: State):
        self.engine = None
        self.projector = None
        self.set_lifecycle_state('unconfigured')
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State):
        if self.active:
            self.on_deactivate(state)
        self.set_lifecycle_state('finalized')
        return TransitionCallbackReturn.SUCCESS

    def set_lifecycle_state(self, name):
        self.lifecycle_state = name
        self.publish_status()

    # ------------------------------------------------------------------ requests
    def on_start_detection(self, request, response):
        try:
            rq.check_name(request.name)
            config = rq.engine_config(request, self.defaults())
            outputs = rq.outputs_of(request.outputs)
            namespace = rq.namespace_of(request.output_namespace, request.name,
                                        self.p('default_namespace'))
            if outputs & rq.OUTPUT_DETECTIONS_3D and self.projector is None:
                outputs &= ~rq.OUTPUT_DETECTIONS_3D
            with self.lock:
                rq.check_conflicts(self.requests, request.name, rq.DetectionRequest,
                                   namespace=namespace)
                message = self.engine.open(request.name, config)
                current = self.requests.get(request.name)
                if current is not None and (current.namespace, current.outputs) == \
                        (namespace, outputs):
                    current.config = config
                    detection = current
                else:
                    detection = rq.DetectionRequest(request.name, config, outputs, namespace)
                    if current is not None:
                        detection.frames = current.frames
                        self.destroy_publishers(current)
                    self.create_publishers(detection)
                    self.requests[request.name] = detection
        except Exception as exc:  # noqa: B902 - the reason goes back to the client
            response.success = False
            response.message = str(exc)
            self.get_logger().warning(f'Detection request {request.name} refused: {exc}')
            return response
        topics = rq.topics_of(detection.namespace, detection.outputs)
        response.success = True
        response.message = message
        response.detections_2d_topic = topics.get(rq.OUTPUT_DETECTIONS_2D, '')
        response.detections_3d_topic = topics.get(rq.OUTPUT_DETECTIONS_3D, '')
        response.detections_image_topic = topics.get(rq.OUTPUT_DETECTIONS_IMAGE, '')
        self.get_logger().info(f'Detection request {request.name}: {message} '
                               f'({detection.namespace or "/"})')
        self.publish_status()
        return response

    def create_publishers(self, detection):
        types = {rq.OUTPUT_DETECTIONS_2D: Detection2DArray,
                 rq.OUTPUT_DETECTIONS_3D: Detection3DArray,
                 rq.OUTPUT_DETECTIONS_IMAGE: Image, rq.OUTPUT_DEBUG_IMAGE: Image}
        for flag, topic in rq.topics_of(detection.namespace, detection.outputs).items():
            detection.publishers[flag] = self.create_publisher(types[flag], topic, 10)
        if detection.outputs & rq.OUTPUT_DETECTIONS_3D:
            detection.publishers['markers'] = self.create_publisher(
                MarkerArray, f'{detection.namespace}/detection_centroid_markers', 10)

    def destroy_publishers(self, request):
        for publisher in request.publishers.values():
            self.destroy_publisher(publisher)
        request.publishers = {}

    def on_start_target_frame(self, request, response):
        try:
            rq.check_name(request.name)
            if not request.frame_id:
                raise rq.RequestError('frame_id is empty')
            if request.mode not in (rq.MODE_SINGLE, rq.MODE_GROUP, rq.MODE_EACH):
                raise rq.RequestError(f'Invalid mode {request.mode}: 0 (single), 1 (group) or '
                                      '2 (each)')
            if request.mode == rq.MODE_SINGLE and not request.id:
                raise rq.RequestError('MODE_SINGLE needs the id of the detection')
            with self.lock:
                source = self.requests.get(request.detection_request)
                if not isinstance(source, rq.DetectionRequest) or source.internal:
                    raise rq.RequestError(
                        f'No open detection request named "{request.detection_request}"')
                if self.projector is None:
                    raise rq.RequestError('The perception system places no detection in 3D '
                                          '(projection.estimator is none)')
                if not source.outputs & rq.OUTPUT_DETECTIONS_3D:
                    raise rq.RequestError(f'The detection request "{source.name}" does not '
                                          'publish detections_3d')
                rq.check_conflicts(self.requests, request.name, rq.FrameRequest,
                                   frame_id=request.frame_id)
                if request.name in self.requests:
                    self.close_request(request.name)
                frame = self.open_frame(request, source)
        except Exception as exc:  # noqa: B902 - the reason goes back to the client
            response.success = False
            response.message = str(exc)
            self.get_logger().warning(f'Target frame request {request.name} refused: {exc}')
            return response
        response.success = True
        response.message = f'Publishing {frame.describe()}'
        self.get_logger().info(f'Target frame request {request.name}: {response.message}')
        self.publish_status()
        return response

    def open_frame(self, request, source):
        namespace = f'{self.p("frames_namespace").rstrip("/")}/{request.name}'
        frame = rq.FrameRequest(
            request.name, source.name, request.frame_id,
            request.fixed_frame_id or self.p('fixed_frame_id'), request.mode,
            request.class_name, request.id, request.n_detections,
            request.half_width_topic or f'{namespace}/half_width')
        if frame.mode == rq.MODE_SINGLE:
            frame.target = targets.SingleTarget(frame.class_name, frame.detection_id)
        elif frame.mode == rq.MODE_EACH:
            frame.target = targets.EachTarget(frame.class_name)
        else:
            frame.target = targets.GroupTarget(frame.class_name, frame.n_detections)
            frame.publishers = {
                'half_width': self.create_publisher(Float64, frame.half_width_topic, 10),
                'tracked_ids': self.create_publisher(Detection2DArray,
                                                     f'{namespace}/tracked_ids', 10),
                'point': self.create_publisher(PointStamped, f'{namespace}/point', 10),
            }
        frame.timer = self.create_timer(self.p('frame_period_ms') / 1000.0,
                                        lambda: self.publish_frame(frame),
                                        callback_group=MutuallyExclusiveCallbackGroup())
        source.frames.add(frame.name)
        self.requests[frame.name] = frame
        if frame.mode == rq.MODE_GROUP:
            self.publish_tracked_ids(frame)
        return frame

    def on_start_skill(self, request, response):
        try:
            rq.check_name(request.name)
            if request.skill not in self.skills.skills:
                raise rq.RequestError(f'Unknown skill "{request.skill}": one of '
                                      f'{", ".join(self.skills.skills)}')
            with self.lock:
                rq.check_conflicts(self.requests, request.name, rq.SkillRequest)
                current = self.requests.get(request.name)
            if current is not None and current.skill != request.skill:
                self.stop(current.name)
            ok, message = self.skills.activate(request.skill)
            if not ok:
                raise rq.RequestError(message)
            with self.lock:
                self.requests[request.name] = rq.SkillRequest(request.name, request.skill)
        except Exception as exc:  # noqa: B902 - the reason goes back to the client
            response.success = False
            response.message = str(exc)
            self.get_logger().warning(f'Skill request {request.name} refused: {exc}')
            return response
        response.success = True
        response.message = message
        response.topics = self.skills.skills[request.skill].topics
        self.get_logger().info(f'Skill request {request.name}: {message}')
        self.publish_status()
        return response

    def on_stop(self, request, response):
        response.success, response.message = self.stop(request.name)
        self.get_logger().info(f'Stop {request.name}: {response.message}')
        self.publish_status()
        return response

    def stop(self, name):
        skill = None
        with self.lock:
            current = self.requests.get(name)
            if current is None or getattr(current, 'internal', False):
                return False, f'No open request named "{name}"'
            ended = self.close_request(name)
            if isinstance(current, rq.SkillRequest) and not any(
                    isinstance(r, rq.SkillRequest) and r.skill == current.skill
                    for r in self.requests.values()):
                skill = current.skill
        if skill is not None:
            self.skills.deactivate(skill)
        return True, f'Ended: {", ".join(ended)}'

    def close_request(self, name):
        """Ends a request (and the frames that follow a detection request); the lock is held.
        Returns the names ended."""
        request = self.requests.pop(name, None)
        if request is None:
            return []
        ended = [name]
        if isinstance(request, rq.DetectionRequest):
            for frame in list(request.frames):
                ended += self.close_request(frame)
            self.engine.close(name)
            self.destroy_publishers(request)
        elif isinstance(request, rq.FrameRequest):
            self.destroy_timer(request.timer)
            if request.mode == rq.MODE_GROUP:
                request.target.reset()
                self.publish_tracked_ids(request)
            self.destroy_publishers(request)
            source = self.requests.get(request.source)
            if source is not None:
                source.frames.discard(name)
        elif isinstance(request, rq.SkillRequest) and not self.active:
            # Deactivating the system: the skill node goes down with it
            if not any(isinstance(r, rq.SkillRequest) and r.skill == request.skill
                       for r in self.requests.values()):
                threading.Thread(target=self.skills.deactivate, args=(request.skill,),
                                 daemon=True).start()
        return ended

    def on_detect_once(self, request, response):
        event = threading.Event()
        with self.lock:
            self.once_count += 1
            name = f'detect_once_{self.once_count}'
        try:
            defaults = self.defaults()
            config = RequestConfig(request.model or defaults['model'], defaults['model_options']
                                   if not request.model else {}, 'none', request.class_name,
                                   request.category, list(request.labels))
            if not config.has_class():
                raise rq.RequestError('No class to detect')
            with self.lock:
                self.engine.open(name, config)
                self.requests[name] = rq.DetectionRequest(
                    name, config, rq.OUTPUT_DETECTIONS_2D | rq.OUTPUT_DETECTIONS_3D, '',
                    internal=True)
                self.once_waiters[name] = [event, None]
            timeout = request.timeout_s if request.timeout_s > 0.0 else 5.0
            if not event.wait(timeout):
                raise rq.RequestError(f'No image processed within {timeout:g} s')
            result = self.once_waiters[name][1]
            if isinstance(result, Exception):
                raise rq.RequestError(f'Detection failed: {result}')
            response.detections_2d, detections_3d = result
            if detections_3d is not None:
                response.detections_3d = detections_3d
            response.success = True
            response.message = f'{len(response.detections_2d.detections)} detections'
        except Exception as exc:  # noqa: B902 - the reason goes back to the client
            response.success = False
            response.message = str(exc)
        finally:
            with self.lock:
                self.once_waiters.pop(name, None)
                if name in self.requests:
                    self.close_request(name)
        return response

    # ------------------------------------------------------------------ images
    def on_camera_info(self, msg):
        if self.projector is not None and self.projector.set_camera_info(msg):
            i = self.projector.intrinsics
            self.get_logger().info(f'Camera calibration (CameraInfo): fx={i.fx:.2f} '
                                   f'fy={i.fy:.2f} cx={i.cx:.2f} cy={i.cy:.2f}')

    def on_image(self, msg):
        if not self.active:
            return
        with self.lock:
            names = [n for n, r in self.requests.items() if isinstance(r, rq.DetectionRequest)]
        if not names:
            return
        start = time.monotonic()
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        results = self.engine.process(image, names)
        detected = False
        with self.lock:
            for name in names:
                request = self.requests.get(name)
                if request is None or name not in results:
                    continue
                result = results[name]
                if isinstance(result, Exception):
                    self.failures += 1
                    self.get_logger().warning(f'Detection failed ({name}): {result}',
                                           throttle_duration_sec=5.0)
                    # No detections: the clients see the target as lost until it works again
                    self.fulfil_once(name, result)
                    self.publish_request(request, msg.header, [], None)
                    continue
                detected = detected or request.config.has_class()
                self.publish_request(request, msg.header, result, image)
        if detected:
            elapsed_ms = (time.monotonic() - start) * 1000.0
            if self.frame_time_pub is not None:
                self.frame_time_pub.publish(Float32(data=elapsed_ms))
            self.frame_times.append(elapsed_ms)
            self.log_stats()

    def publish_request(self, request, header, detections, image):
        """The outputs of a request for one image (the lock is held). image None: the image
        failed, only the empty detections are published."""
        detections_2d = detections_message(header, detections)
        pubs = request.publishers
        if rq.OUTPUT_DETECTIONS_2D in pubs:
            pubs[rq.OUTPUT_DETECTIONS_2D].publish(detections_2d)
        detections_3d, placed = None, []
        if self.projector is not None and (rq.OUTPUT_DETECTIONS_3D in pubs or request.frames or
                                           request.internal):
            # Also for an empty message: the clients and the target frames see the target lost
            detections_3d, placed = self.projector.project(detections_2d, image)
        if detections_3d is not None:
            request.last = (detections_2d.header, detections,
                            dict(zip(placed, detections_3d.detections)))
        else:
            request.last = (detections_2d.header, detections, {})
        if detections_3d is not None and rq.OUTPUT_DETECTIONS_3D in pubs:
            pubs[rq.OUTPUT_DETECTIONS_3D].publish(detections_3d)
            pubs['markers'].publish(self.projector.markers(detections_3d, request.markers))
            request.markers = len(detections_3d.detections)
        if detections_3d is not None:
            for frame_name in request.frames:
                self.update_frame(self.requests[frame_name], detections_3d)
        if request.internal:
            self.fulfil_once(request.name, (detections_2d, detections_3d))
            return
        if image is None:
            return
        if rq.OUTPUT_DETECTIONS_IMAGE in pubs:
            marked_image = marking.marked_image(image, detections)
            if self.p('zoom_small_detections'):
                marked_image = marking.with_zoomed_detections(image, marked_image, detections)
            marked = self.bridge.cv2_to_imgmsg(marked_image, 'rgb8')
            marked.header = header
            pubs[rq.OUTPUT_DETECTIONS_IMAGE].publish(marked)
        if rq.OUTPUT_DEBUG_IMAGE in pubs:
            debug = self.bridge.cv2_to_imgmsg(marking.debug_image(image, detections), 'bgr8')
            debug.header = header
            pubs[rq.OUTPUT_DEBUG_IMAGE].publish(debug)

    def fulfil_once(self, name, result):
        waiter = self.once_waiters.get(name)
        if waiter is not None and waiter[1] is None:
            waiter[1] = result
            waiter[0].set()

    def log_stats(self):
        elapsed = time.monotonic() - self.stats_start
        if elapsed < self.p('stats_period_s') or not self.frame_times:
            return
        self.stats = (len(self.frame_times) / elapsed, statistics.median(self.frame_times),
                      self.failures)
        self.get_logger().info(
            f'{self.stats[0]:.1f} images/s, {self.stats[1]:.0f} ms per image (median), '
            f'{self.failures} failures in {elapsed:.0f} s')
        self.frame_times, self.failures, self.stats_start = [], 0, time.monotonic()

    # ------------------------------------------------------------------ target frames
    def update_frame(self, frame, detections_3d):
        log = self.get_logger().debug
        frame.target.update(detections_3d, log)
        if frame.mode == rq.MODE_GROUP and frame.target.half_width is not None:
            frame.publishers['half_width'].publish(Float64(data=frame.target.half_width))

    def camera_pose(self, fixed_frame_id, camera_frame_id, stamp_msg, timeout_ms):
        """The transform fixed <- camera at the time of an image, or None (logged)."""
        if not camera_frame_id:
            self.get_logger().warning('Optical frame id not obtained', throttle_duration_sec=5.0)
            return None
        stamp = Time.from_msg(stamp_msg)
        timeout = Duration(nanoseconds=int(timeout_ms) * 1_000_000)
        try:
            # Camera pose when the image was captured, not the latest one: while the robot turns,
            # the latest pose would shift the detection in the turning direction
            if not self.tf_buffer.can_transform(fixed_frame_id, camera_frame_id, stamp, timeout):
                raise tf2_ros.TransformException('not available')
            return self.tf_buffer.lookup_transform(fixed_frame_id, camera_frame_id,
                                                   stamp).transform
        except tf2_ros.TransformException as exc:
            self.get_logger().warning(f'Error in TF {fixed_frame_id} -> {camera_frame_id} '
                                   f'[{exc}]', throttle_duration_sec=2.0)
            return None

    def publish_frame(self, frame):
        with self.lock:
            if self.requests.get(frame.name) is not frame:
                return
            taken = frame.target.take()
        if not taken:
            return
        # (child frame, point): one for a single target or a group, one per detection for each
        points = [(f'{frame.frame_id}_{i}', p) for i, p in taken] \
            if frame.mode == rq.MODE_EACH else [(frame.frame_id, taken)]
        point = points[0][1]
        fixed_from_camera = self.camera_pose(frame.fixed_frame_id, point.frame_id, point.stamp,
                                             self.p('tf_lookup_timeout_ms'))
        if fixed_from_camera is None:
            return
        transforms_out = []
        for child, point in points:
            x, y, z = transforms.apply(fixed_from_camera, (point.x, point.y, point.z))
            transform = TransformStamped()
            # The time of the image: where the target was then (the attention system stamped it
            # with the time of publication)
            transform.header.stamp = point.stamp
            transform.header.frame_id = frame.fixed_frame_id
            transform.child_frame_id = child
            transform.transform.translation.x = x
            transform.transform.translation.y = y
            transform.transform.translation.z = z
            transform.transform.rotation.w = 1.0
            transforms_out.append(transform)
        transform = transforms_out[0]
        with self.lock:
            if self.requests.get(frame.name) is not frame:
                return
            self.tf_broadcaster.sendTransform(transforms_out)
            if frame.mode == rq.MODE_GROUP:
                frame.publishers['point'].publish(PointStamped(
                    header=transform.header, point=transform.transform.translation))
                self.publish_tracked_ids(frame, point.frame_id)

    def publish_tracked_ids(self, frame, optical_frame_id=''):
        publisher = frame.publishers.get('tracked_ids')
        if publisher is None:
            return
        msg = Detection2DArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = optical_frame_id
        for tracked_id in frame.target.tracked_ids:
            detection = Detection2D(header=msg.header, id=tracked_id)
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = frame.class_name
            detection.results.append(hypothesis)
            msg.detections.append(detection)
        publisher.publish(msg)

    # ------------------------------------------------------------------ objects and models
    def on_get_objects(self, request, response):
        with self.lock:
            detection = self.requests.get(request.request)
            if not isinstance(detection, rq.DetectionRequest) or detection.internal:
                response.success = False
                response.message = f'No open detection request named "{request.request}"'
                return response
            last = detection.last
        if last is None:
            response.success = False
            response.message = 'No image processed yet'
            return response
        header, detections, placed = last
        response.stamp = header.stamp
        fixed = self.p('fixed_frame_id')
        pose = self.camera_pose(fixed, header.frame_id, header.stamp, 200) if placed else None
        for index, det in enumerate(detections):
            if request.only_tracked and not det.id:
                continue
            obj = TrackedObject(id=det.id, class_name=det.label, score=float(det.score))
            obj.bbox.center.position.x, obj.bbox.center.position.y = det.cx, det.cy
            obj.bbox.size_x, obj.bbox.size_y = det.w, det.h
            if index in placed and pose is not None:
                p = placed[index].bbox.center.position
                x, y, z = transforms.apply(pose, (p.x, p.y, p.z))
                obj.has_position = True
                obj.position.header.stamp, obj.position.header.frame_id = header.stamp, fixed
                obj.position.point.x, obj.position.point.y, obj.position.point.z = x, y, z
            response.objects.append(obj)
        response.success = True
        response.message = f'{len(response.objects)} objects'
        return response

    def on_list_models(self, request, response):
        response.models = self.engine.available()
        response.loaded = self.engine.models()
        response.default_model = self.p('model')
        model = request.model or response.default_model
        try:
            options = self.defaults()['model_options'] if model == self.p('model') else {}
            response.open_vocabulary, classes = self.engine.model_info(model, options)
            response.classes = list(classes)
        except Exception as exc:  # noqa: B902 - an unknown model: no classes
            self.get_logger().warning(f'list_models {model}: {exc}')
        return response

    # ------------------------------------------------------------------ user and status
    def on_emotion(self, msg):
        import json
        try:
            data = json.loads(msg.data)
        except ValueError as exc:
            self.get_logger().warning(f'Could not parse the emotion: {exc}')
            return
        with self.lock:
            self.user_state.stamp = self.get_clock().now().to_msg()
            self.user_state.emotion = str(data.get('emotion', 'neutral'))
            self.user_state.confidence = float(data.get('confidence', 0.0))
            self.user_state.source = str(data.get('source', 'vision'))
            self.user_state_available = True

    def on_speaking(self, msg):
        with self.lock:
            self.user_state.speaking = msg.data

    def on_get_user_state(self, request, response):
        with self.lock:
            response.available = self.user_state_available
            response.state = self.user_state
        return response

    def status(self):
        msg = PerceptionStatus()
        msg.stamp = self.get_clock().now().to_msg()
        msg.state = self.lifecycle_state
        with self.lock:
            for name, request in sorted(self.requests.items()):
                if getattr(request, 'internal', False):
                    continue
                item = RequestStatus(name=name, description=request.describe(), ok=True)
                if isinstance(request, rq.DetectionRequest):
                    item.kind = RequestStatus.KIND_DETECTION
                    item.outputs = list(rq.topics_of(request.namespace,
                                                     request.outputs).values())
                elif isinstance(request, rq.FrameRequest):
                    item.kind = RequestStatus.KIND_TARGET_FRAME
                    item.outputs = [request.frame_id]
                    item.ok = request.target.seen
                else:
                    item.kind = RequestStatus.KIND_SKILL
                    item.outputs = self.skills.skills[request.skill].topics
                msg.requests.append(item)
            msg.images_per_second, msg.ms_per_image, msg.failures = self.stats
            msg.models = self.engine.models() if self.engine is not None else []
            msg.active_skills = sorted(self.skills.active) if self.skills is not None else []
        return msg

    def publish_status(self):
        self.status_pub.publish(self.status())

    def on_get_status(self, request, response):
        response.status = self.status()
        return response


def detections_message(header, detections):
    """The Detection2DArray of a request: the stamp and frame of the image (the clients need the
    camera pose at capture time), the class in results[0], the tracker id in id."""
    out = Detection2DArray(header=Header(stamp=header.stamp, frame_id=header.frame_id))
    for det in detections:
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = det.label
        hypothesis.hypothesis.score = det.score
        d = Detection2D(header=out.header, id=det.id)
        d.bbox.center.position.x = det.cx
        d.bbox.center.position.y = det.cy
        d.bbox.size_x = det.w
        d.bbox.size_y = det.h
        d.results.append(hypothesis)
        out.detections.append(d)
    return out


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionSystem()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    if node.get_parameter('autostart').value:
        # Configure and activate once the executor runs (the transitions may call services)
        def start():
            timer.cancel()
            if node.trigger_configure() == TransitionCallbackReturn.SUCCESS:
                node.trigger_activate()
        timer = node.create_timer(0.1, start, callback_group=MutuallyExclusiveCallbackGroup())
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:  # noqa: B902 - the context shut down by a signal while spinning
        if rclpy.ok():
            raise
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
