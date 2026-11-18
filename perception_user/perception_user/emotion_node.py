# Copyright 2026 Álvaro Quintana Camacho
# Copyright 2026 Rodrigo Pérez-Rodríguez
#
# Licensed under the MIT License (LICENSE.MIT of this package). Ported from the emotion detector
# of interaction_system (interaction_perception), as a lifecycle node.

"""emotion_detector: the facial expression of the user (the skill "emotion" of the perception
system). EXPERIMENTAL, NOT VALIDATED.

The largest face of each image (MediaPipe FaceDetector) is classified by a Hugging Face image
classifier, and the label is published as JSON. Each image is classified on its own, with a
third-party model trained on posed faces; a facial expression is not a reliable sign of an
emotion, and the EU AI Act restricts emotion recognition. See the limitations in
docs/USER_PERCEPTION.md before using it with people.

  input    image_topic (sensor_msgs/Image)
  output   output_topic (std_msgs/String, /user_emotion): {"emotion", "confidence", "source"}
"""

import json
import threading

from cv_bridge import CvBridge
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from perception_user.models import model_file

FACE_DETECTOR_URL = ('https://storage.googleapis.com/mediapipe-models/face_detector/'
                     'blaze_face_short_range/float16/1/blaze_face_short_range.tflite')
IMAGE_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                       history=HistoryPolicy.KEEP_LAST, depth=1)


def largest_face(detections, width, height):
    """(x, y, w, h) in pixels of the largest face, or None."""
    best = None
    for detection in detections:
        box = detection.bounding_box
        x, y = max(0, int(box.origin_x)), max(0, int(box.origin_y))
        w, h = min(int(box.width), width - x), min(int(box.height), height - y)
        if w > 0 and h > 0 and (best is None or w * h > best[2] * best[3]):
            best = (x, y, w, h)
    return best


class EmotionDetector(LifecycleNode):

    def __init__(self):
        super().__init__('emotion_detector')
        self.declare_parameters('', [
            ('image_topic', '/image_rgb'),
            ('output_topic', '/user_emotion'),
            ('model_name', 'dima806/face_emotions_image_detection'),
            ('min_detection_confidence', 0.5),
        ])
        self.bridge = CvBridge()
        self.face_detector = None
        self.classifier = None
        self.busy = threading.Lock()
        self.image_sub = None
        self.publisher = None

    def p(self, name):
        return self.get_parameter(name).value

    def on_configure(self, state: State):
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            from transformers import pipeline
            options = mp_vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=model_file(FACE_DETECTOR_URL, self.get_logger().info)),
                min_detection_confidence=float(self.p('min_detection_confidence')))
            self.face_detector = mp_vision.FaceDetector.create_from_options(options)
            self.classifier = pipeline('image-classification', model=self.p('model_name'),
                                       device=-1)
        except Exception as exc:  # noqa: B902 - any failure fails the transition
            self.get_logger().error(f'Cannot load the emotion models: {exc}')
            return TransitionCallbackReturn.FAILURE
        self.publisher = self.create_lifecycle_publisher(String, self.p('output_topic'), 10)
        self.get_logger().warning('The emotion detector is experimental and not validated')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State):
        self.image_sub = self.create_subscription(Image, self.p('image_topic'), self.on_image,
                                                  IMAGE_QOS)
        return super().on_activate(state)

    def on_deactivate(self, state: State):
        if self.image_sub is not None:
            self.destroy_subscription(self.image_sub)
            self.image_sub = None
        return super().on_deactivate(state)

    def on_cleanup(self, state: State):
        if self.face_detector is not None:
            self.face_detector.close()
        self.face_detector = None
        self.classifier = None
        self.destroy_lifecycle_publisher(self.publisher)
        return TransitionCallbackReturn.SUCCESS

    def on_image(self, msg):
        # One classification at a time, in its own thread: the images that arrive meanwhile are
        # skipped
        if self.busy.locked():
            return
        rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        threading.Thread(target=self.classify, args=(rgb,), daemon=True).start()

    def classify(self, rgb):
        import cv2
        import mediapipe as mp
        from PIL import Image as PILImage
        if not self.busy.acquire(blocking=False):
            return
        try:
            result = self.face_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                        data=rgb))
            face = largest_face(result.detections, rgb.shape[1], rgb.shape[0])
            if face is None:
                return
            x, y, w, h = face
            crop = cv2.resize(rgb[y:y + h, x:x + w], (224, 224))
            prediction = self.classifier(PILImage.fromarray(crop))
            if prediction:
                self.publisher.publish(String(data=json.dumps({
                    'emotion': prediction[0]['label'].lower(),
                    'confidence': round(float(prediction[0]['score']), 3),
                    'source': 'vision'})))
        except Exception as exc:  # noqa: B902 - a failed image is skipped
            self.get_logger().error(f'Emotion classification failed: {exc}',
                                    throttle_duration_sec=5.0)
        finally:
            self.busy.release()


def main(args=None):
    rclpy.init(args=args)
    node = EmotionDetector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
