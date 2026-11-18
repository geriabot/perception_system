# Copyright 2026 Álvaro Quintana Camacho
# Copyright 2026 Rodrigo Pérez-Rodríguez
#
# Licensed under the MIT License (LICENSE.MIT of this package). Ported from the lip activity
# detector of interaction_system (interaction_perception), as a lifecycle node of the perception
# system; the classification is unchanged.

"""lip_activity_detector: whether the user is speaking, from the movement of the lips (the skill
"lip_activity" of the perception system).

MediaPipe FaceLandmarker gives the lip gap of the face, normalised by the height of the face. The
detector computes the variance of the consecutive changes of the gap (roughness) over a short
window: speech oscillates fast and keeps it high, while a single opening, a yawn or a smile moves
monotonically and it drops as soon as the mouth stops. A continuous onset is needed before it
reports speaking, and a continuous calm mouth before it stops.

A lifecycle node: the face model is loaded on configure, and the images are processed only while
it is active (the perception system activates it while a client needs the skill).

  input    image_topic (sensor_msgs/Image)
  outputs  speaking_topic (std_msgs/Bool, /user_speaking): on every change, and False when the
           node is deactivated while speaking
           lip_activity_topic (std_msgs/Float32, /lip_activity): the lip gap ratio of each image
"""

import collections
import time

import numpy as np

UPPER_LIP = 13
LOWER_LIP = 14
NOSE_TIP = 4
CHIN = 152

MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/'
             'float16/1/face_landmarker.task')


class LipFSM:
    """SILENT -> SPEAKING: the roughness (variance of consecutive lip-gap differences over the last
    window_frames) stays at or above roughness_threshold for onset_secs.
    SPEAKING -> SILENT: it stays below for silence_secs."""

    SILENT = 'SILENT'
    SPEAKING = 'SPEAKING'

    def __init__(self, roughness_threshold, onset_secs, window_frames, silence_secs,
                 clock=time.monotonic):
        self.roughness_threshold = roughness_threshold
        self.onset_secs = onset_secs
        self.silence_secs = silence_secs
        self.clock = clock
        self.state = self.SILENT
        self.buffer = collections.deque(maxlen=window_frames)
        self.active_since = None
        self.below_since = None
        self.last_roughness = 0.0

    def update(self, ratio):
        """Feed the smoothed lip-gap ratio; returns (state, changed)."""
        now = self.clock()
        previous = self.state
        self.buffer.append(ratio)
        self.last_roughness = float(np.var(np.diff(np.array(self.buffer)))) \
            if len(self.buffer) >= 4 else 0.0
        active = self.last_roughness >= self.roughness_threshold
        if self.state == self.SILENT:
            if active:
                if self.active_since is None:
                    self.active_since = now
                elif now - self.active_since >= self.onset_secs:
                    self.state = self.SPEAKING
                    self.below_since = None
            else:
                self.active_since = None
        else:
            if not active:
                if self.below_since is None:
                    self.below_since = now
                elif now - self.below_since >= self.silence_secs:
                    self.state = self.SILENT
                    self.active_since = None
            else:
                self.below_since = None
        return self.state, self.state != previous


def mouth_openness(landmarks):
    """The lip gap normalised by the height of the face (nose tip to chin)."""
    lip_gap = abs(landmarks[LOWER_LIP].y - landmarks[UPPER_LIP].y)
    face_height = abs(landmarks[CHIN].y - landmarks[NOSE_TIP].y) + 1e-9
    return float(lip_gap / face_height)


class LipActivity:
    """The classification of a stream of lip-gap ratios (no ROS, no model): smoothing, then
    LipFSM. ratio 0.0: no face in the image."""

    def __init__(self, smoothing_frames=4, roughness_threshold=7e-6, roughness_window=10,
                 onset_secs=0.9, silence_secs=1.2, clock=time.monotonic):
        self.window = collections.deque(maxlen=smoothing_frames)
        self.fsm = LipFSM(roughness_threshold, onset_secs, roughness_window, silence_secs, clock)

    def update(self, ratio):
        """Returns (speaking, changed, smoothed ratio)."""
        self.window.append(ratio)
        smoothed = float(np.mean(self.window))
        state, changed = self.fsm.update(smoothed)
        return state == LipFSM.SPEAKING, changed, smoothed
