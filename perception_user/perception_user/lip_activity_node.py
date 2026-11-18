# Copyright 2026 Álvaro Quintana Camacho
# Copyright 2026 Rodrigo Pérez-Rodríguez
#
# Licensed under the MIT License (LICENSE.MIT of this package). Ported from the lip activity
# detector of interaction_system (interaction_perception), as a lifecycle node.

"""The node /lip_activity_detector (see lip_activity.py)."""

from cv_bridge import CvBridge
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32

from perception_user.lip_activity import LipActivity, MODEL_URL, mouth_openness
from perception_user.models import model_file, without_audio

IMAGE_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                       history=HistoryPolicy.KEEP_LAST, depth=1)


class LipActivityDetector(LifecycleNode):

    def __init__(self):
        super().__init__('lip_activity_detector')
        self.declare_parameters('', [
            ('image_topic', '/image_rgb'),
            ('speaking_topic', '/user_speaking'),
            ('lip_activity_topic', '/lip_activity'),
            ('min_detection_confidence', 0.5),
            ('min_tracking_confidence', 0.5),
            # Light smoothing: removes single-frame spikes without killing the speech
            # oscillations (~5 Hz). Above 6 frames the speech signal is suppressed
            ('smoothing_frames', 4),
            # Variance of consecutive lip-gap differences above which the lips move as in speech.
            # Real speech: ~1e-5..1e-4. Yawn, smile, a single opening: ~1e-8..1e-7
            ('roughness_threshold', 7e-6),
            # Short window (~0.33 s at 30 images/s): a brief event leaves it quickly
            ('roughness_window', 10),
            # Continuous time with high roughness to report speaking: a quick open-close
            # (~0.5 s of movement, ~0.83 s with the window) does not reach it
            ('onset_secs', 0.9),
            # Continuous time with a calm mouth to report silence
            ('silence_secs', 1.2),
        ])
        self.bridge = CvBridge()
        self.landmarker = None
        self.activity = None
        self.speaking = False
        self.timestamp_ms = 0
        self.image_sub = None
        self.speaking_pub = None
        self.ratio_pub = None

    def p(self, name):
        return self.get_parameter(name).value

    def on_configure(self, state: State):
        try:
            without_audio()
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            options = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=model_file(MODEL_URL, self.get_logger().info)),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=float(self.p('min_detection_confidence')),
                min_tracking_confidence=float(self.p('min_tracking_confidence')))
            self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        except Exception as exc:  # noqa: B902 - any failure fails the transition
            self.get_logger().error(f'Cannot load the face landmarker: {exc}')
            return TransitionCallbackReturn.FAILURE
        self.speaking_pub = self.create_publisher(Bool, self.p('speaking_topic'), 10)
        self.ratio_pub = self.create_lifecycle_publisher(Float32, self.p('lip_activity_topic'),
                                                         10)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State):
        self.activity = LipActivity(int(self.p('smoothing_frames')),
                                    float(self.p('roughness_threshold')),
                                    int(self.p('roughness_window')),
                                    float(self.p('onset_secs')), float(self.p('silence_secs')))
        self.speaking = False
        self.image_sub = self.create_subscription(Image, self.p('image_topic'), self.on_image,
                                                  IMAGE_QOS)
        self.get_logger().info(f'Lip activity active, images from {self.p("image_topic")}')
        return super().on_activate(state)

    def on_deactivate(self, state: State):
        if self.image_sub is not None:
            self.destroy_subscription(self.image_sub)
            self.image_sub = None
        if self.speaking:
            self.speaking = False
            self.speaking_pub.publish(Bool(data=False))
        return super().on_deactivate(state)

    def on_cleanup(self, state: State):
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None
        self.destroy_publisher(self.speaking_pub)
        self.destroy_lifecycle_publisher(self.ratio_pub)
        return TransitionCallbackReturn.SUCCESS

    def on_image(self, msg):
        import mediapipe as mp
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as exc:  # noqa: B902
            self.get_logger().error(f'cv_bridge error: {exc}', throttle_duration_sec=5.0)
            return
        # VIDEO mode needs increasing timestamps: those of the images, or +33 ms if they repeat
        stamp_ms = msg.header.stamp.sec * 1000 + msg.header.stamp.nanosec // 1_000_000
        self.timestamp_ms = max(stamp_ms, self.timestamp_ms + 1) if stamp_ms > 0 \
            else self.timestamp_ms + 33
        result = self.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self.timestamp_ms)
        ratio = mouth_openness(result.face_landmarks[0]) if result.face_landmarks else 0.0
        self.ratio_pub.publish(Float32(data=float(ratio)))
        speaking, changed, smoothed = self.activity.update(ratio)
        if changed:
            self.speaking = speaking
            self.speaking_pub.publish(Bool(data=speaking))
            self.get_logger().info(
                f'{"SPEAKING" if speaking else "SILENT"}  roughness='
                f'{self.activity.fsm.last_roughness:.2e}  ratio={ratio:.3f}  '
                f'smoothed={smoothed:.3f}')


def run(args=None):
    rclpy.init(args=args)
    node = LipActivityDetector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:  # noqa: B902 - re-raised unless the context has ended
        # A callback that was running when the process was asked to stop (Ctrl+C, the end of a
        # launch) publishes after the context has ended: not an error. Raised if it is not that
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def main(args=None):
    run(args)
