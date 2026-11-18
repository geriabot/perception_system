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

"""What a person's body says, from the 17 keypoints of a pose model (COCO): posture (standing,
sitting, lying), a fall, gestures (a raised hand, both hands raised, waving, pointing), where they
point on the floor, whether they face the camera, and the color of their clothes. No ROS here: the
node gives the keypoints of each image, in pixels and, when it can, in the fixed frame (z up).

The rules are geometric and documented with their thresholds (BodyParameters), so that a robot
can be tuned without retraining anything. In the image, y grows downwards; in the fixed frame,
z grows upwards and the floor is at floor_z. Left and right are the person's.
"""

from collections import deque
from dataclasses import dataclass, field
import math

import numpy as np

KEYPOINT_NAMES = (
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle')
K = {name: i for i, name in enumerate(KEYPOINT_NAMES)}
SIDES = ('left', 'right')

STANDING, SITTING, LYING, UNKNOWN = 'standing', 'sitting', 'lying', 'unknown'
GESTURES = ('hand_raised_left', 'hand_raised_right', 'both_hands_raised', 'waving',
            'pointing_left', 'pointing_right')
# A gesture asked for by a client, and the gestures that satisfy it
GESTURE_QUERIES = {
    'hand_raised': ('hand_raised_left', 'hand_raised_right', 'both_hands_raised'),
    'pointing': ('pointing_left', 'pointing_right'),
}
FALL, RECOVERED = 'fall', 'recovered'

# Sizes of an adult body (m), for the distance when there is no depth image
TORSO_LENGTH_M = 0.50          # the middle of the shoulders to the middle of the hips
SHOULDER_WIDTH_M = 0.36

# The colors of clothes, by the hue of OpenCV (0-180)
COLOR_NAMES = ('red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'black',
               'white', 'gray')


@dataclass
class BodyParameters:
    keypoint_threshold: float = 0.5     # a keypoint with a lower score is not used
    # Posture: the torso more than this from the vertical is lying
    lying_torso_angle_deg: float = 55.0
    # Sitting: the thigh more than this from the vertical, or shorter than this part of the
    # torso in the image (a thigh towards the camera)
    sitting_thigh_angle_deg: float = 50.0
    sitting_thigh_ratio: float = 0.5
    # Without the knees, with 3D: the hips lower than this over the floor is sitting
    sitting_max_hip_height_m: float = 0.7
    posture_hold_s: float = 0.5         # a new posture is reported after this long
    # Gestures: a hand is raised with the wrist this far (in torso lengths) over the shoulder
    hand_raised_margin: float = 0.3
    # Pointing: the elbow this straight, and the arm this far from hanging down
    pointing_elbow_angle_deg: float = 150.0
    pointing_min_elevation_deg: float = 40.0
    pointing_min_arm_ratio: float = 0.6  # the arm in the image, in torso lengths (2D rule)
    # Waving: the hand over the elbow, moving sideways at least waving_amplitude (torso lengths)
    # and changing direction waving_reversals times within waving_window_s
    waving_window_s: float = 1.5
    waving_amplitude: float = 0.12
    waving_reversals: int = 2
    gesture_hold_s: float = 0.4         # a gesture is reported after this long
    gesture_release_s: float = 0.5      # and kept for this long after it is no longer seen
    # A fall: lying for fall_confirm_s, with the torso (if placed in 3D) below fall_max_height_m
    # over the floor; recovered after recover_s not lying
    fall_confirm_s: float = 1.0
    fall_max_height_m: float = 0.6
    recover_s: float = 1.0
    # Where the arm points: the floor (z of the fixed frame), up to this distance from the hand
    floor_z: float = 0.0
    max_pointing_m: float = 8.0
    # Facing the camera: the nose this far from the middle of the eyes (0: at the middle, 0.5:
    # at an eye)
    facing_asymmetry: float = 0.3
    # The color of the clothes: the part of the torso used (from its center)
    torso_crop: float = 0.6
    forget_s: float = 3.0               # a person not seen for this long is forgotten


def _point(kp, name, threshold):
    """The pixel (x, y) of a keypoint, or None if its score is low."""
    x, y, score = kp[K[name]]
    return None if score < threshold else np.array([x, y], dtype=float)


def _mid(a, b):
    if a is None or b is None:
        return a if b is None else b
    return (a + b) / 2.0


def _angle_from(vector, reference):
    """The angle (degrees) between two vectors."""
    norm = np.linalg.norm(vector) * np.linalg.norm(reference)
    if norm < 1e-9:
        return 0.0
    cosine = float(np.clip(np.dot(vector, reference) / norm, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _joint_angle(a, b, c):
    """The angle at b (degrees) of the segments b-a and b-c: 180 is straight."""
    return _angle_from(a - b, c - b)


def torso(kp, threshold):
    """(middle of the shoulders, middle of the hips, torso length) in pixels, or None."""
    shoulders = _mid(_point(kp, 'left_shoulder', threshold),
                     _point(kp, 'right_shoulder', threshold))
    hips = _mid(_point(kp, 'left_hip', threshold), _point(kp, 'right_hip', threshold))
    if shoulders is None or hips is None:
        return None
    length = float(np.linalg.norm(shoulders - hips))
    return (shoulders, hips, length) if length > 1.0 else None


def posture_2d(kp, p, box=None):
    """standing, sitting, lying or unknown, from the keypoints in the image."""
    t = torso(kp, p.keypoint_threshold)
    if t is None:
        return UNKNOWN
    shoulders, hips, length = t
    # Up in the image is -y
    if _angle_from(shoulders - hips, np.array([0.0, -1.0])) > p.lying_torso_angle_deg:
        return LYING
    if box is not None:
        width, height = box[2] - box[0], box[3] - box[1]
        if width > 1.3 * height and \
                _angle_from(shoulders - hips, np.array([0.0, -1.0])) > 40.0:
            return LYING
    thighs = []
    for side in SIDES:
        hip, knee = _point(kp, f'{side}_hip', p.keypoint_threshold), \
            _point(kp, f'{side}_knee', p.keypoint_threshold)
        if hip is not None and knee is not None:
            thighs.append(knee - hip)
    if not thighs:
        return UNKNOWN
    thigh = max(thighs, key=lambda v: _angle_from(v, np.array([0.0, 1.0])))
    angle = _angle_from(thigh, np.array([0.0, 1.0]))
    ratio = max(float(np.linalg.norm(v)) for v in thighs) / length
    if angle > p.sitting_thigh_angle_deg or ratio < p.sitting_thigh_ratio:
        return SITTING
    return STANDING


def posture_3d(points, valid, p):
    """The posture from the keypoints in the fixed frame (z up), or None if the torso is not
    placed. Lying from the torso; sitting or standing from the height of the hips."""
    def mid(a, b):
        found = [points[K[n]] for n in (a, b) if valid[K[n]]]
        return np.mean(found, axis=0) if found else None
    shoulders, hips = mid('left_shoulder', 'right_shoulder'), mid('left_hip', 'right_hip')
    if shoulders is None or hips is None:
        return None
    if _angle_from(shoulders - hips, np.array([0.0, 0.0, 1.0])) > p.lying_torso_angle_deg:
        return LYING
    return SITTING if hips[2] - p.floor_z < p.sitting_max_hip_height_m else STANDING


def raised_hands(kp, p):
    """The sides whose wrist is over the shoulder by hand_raised_margin torso lengths."""
    t = torso(kp, p.keypoint_threshold)
    if t is None:
        return set()
    raised = set()
    for side in SIDES:
        wrist = _point(kp, f'{side}_wrist', p.keypoint_threshold)
        shoulder = _point(kp, f'{side}_shoulder', p.keypoint_threshold)
        if wrist is not None and shoulder is not None and \
                wrist[1] < shoulder[1] - p.hand_raised_margin * t[2]:
            raised.add(side)
    return raised


def pointing_arms(kp, p, points=None, valid=None, depth_3d=False):
    """The sides whose arm points: straight at the elbow and lifted from hanging down, not
    raised over the head. With keypoints placed by a depth image (depth_3d), in 3D, which also
    sees an arm that points at the camera; otherwise in the image."""
    t = torso(kp, p.keypoint_threshold)
    if t is None:
        return set()
    raised = raised_hands(kp, p)
    sides = set()
    for side in SIDES:
        if side in raised:
            continue
        names = [f'{side}_shoulder', f'{side}_elbow', f'{side}_wrist']
        if depth_3d and points is not None and all(valid[K[n]] for n in names):
            shoulder, elbow, wrist = (points[K[n]] for n in names)
            arm = wrist - shoulder
            if np.linalg.norm(arm) < 0.35:
                continue
            down = np.array([0.0, 0.0, -1.0])
        else:
            joints = [_point(kp, n, p.keypoint_threshold) for n in names]
            if any(j is None for j in joints):
                continue
            shoulder, elbow, wrist = joints
            arm = wrist - shoulder
            if np.linalg.norm(arm) < p.pointing_min_arm_ratio * t[2]:
                continue
            down = np.array([0.0, 1.0])
        if _joint_angle(shoulder, elbow, wrist) < p.pointing_elbow_angle_deg:
            continue
        if _angle_from(arm, down) < p.pointing_min_elevation_deg:
            continue
        sides.add(side)
    return sides


def pointed_location(shoulder, wrist, p):
    """Where the line from the shoulder through the wrist (fixed frame) meets the floor, and the
    unit direction of the arm: (direction, point or None)."""
    arm = np.asarray(wrist, dtype=float) - np.asarray(shoulder, dtype=float)
    norm = np.linalg.norm(arm)
    if norm < 1e-6:
        return None, None
    direction = arm / norm
    if direction[2] > -0.02:    # level or upwards: it does not meet the floor
        return direction, None
    distance = (np.asarray(wrist)[2] - p.floor_z) / -direction[2]
    if distance > p.max_pointing_m:
        return direction, None
    return direction, np.asarray(wrist, dtype=float) + distance * direction


def facing_camera(kp, p):
    """The face is turned to the camera: both eyes and the nose seen, and the nose near the
    middle of the eyes (within facing_asymmetry of half their distance from it). A face turned
    aside has its nose near one eye, or one eye hidden."""
    thr = p.keypoint_threshold
    nose, left_eye, right_eye = (_point(kp, n, thr) for n in ('nose', 'left_eye', 'right_eye'))
    if nose is None or left_eye is None or right_eye is None:
        return False
    width = left_eye[0] - right_eye[0]
    if abs(width) < 1.0:
        return False
    return bool(abs((nose[0] - right_eye[0]) / width - 0.5) <= p.facing_asymmetry)


def head_point(points, valid):
    """The head in 3D: the nose, else the middle of the eyes, else of the ears; None if unseen."""
    for names in (('nose',), ('left_eye', 'right_eye'), ('left_ear', 'right_ear')):
        found = [points[K[n]] for n in names if valid[K[n]]]
        if found:
            return np.mean(found, axis=0)
    return None


def torso_point(points, valid):
    found = [points[K[n]] for n in ('left_shoulder', 'right_shoulder', 'left_hip', 'right_hip')
             if valid[K[n]]]
    return np.mean(found, axis=0) if found else None


def color_name(hsv):
    """The name of the color of HSV pixels (OpenCV ranges: H 0-180, S and V 0-255), per pixel."""
    h, s, v = (hsv[..., i].astype(int) for i in range(3))
    names = np.full(h.shape, 'gray', dtype=object)
    chromatic = (s >= 64) & (v >= 51)
    hue_names = [(8, 'red'), (20, 'orange'), (33, 'yellow'), (85, 'green'), (128, 'blue'),
                 (150, 'purple'), (165, 'pink'), (181, 'red')]
    lower = 0
    for upper, name in hue_names:
        names[chromatic & (h >= lower) & (h < upper)] = name
        lower = upper
    names[chromatic & (h < 20) & (v < 140)] = 'brown'
    names[~chromatic & (v >= 191)] = 'white'
    names[v < 51] = 'black'
    return names


def clothing_color(image, kp, p):
    """The dominant color of the upper body (the central part of the quadrilateral of the
    shoulders and hips; without the hips, a square under the shoulders), or '' if the shoulders
    are not seen. image: RGB uint8."""
    import cv2
    thr = p.keypoint_threshold
    left, right = _point(kp, 'left_shoulder', thr), _point(kp, 'right_shoulder', thr)
    if left is None or right is None:
        return ''
    left_hip, right_hip = _point(kp, 'left_hip', thr), _point(kp, 'right_hip', thr)
    if left_hip is None or right_hip is None:
        down = np.array([0.0, float(np.linalg.norm(left - right))])
        left_hip, right_hip = left + down, right + down
    corners = np.array([left, right, right_hip, left_hip])
    center = corners.mean(axis=0)
    polygon = (center + (corners - center) * p.torso_crop).astype(np.int32)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 1)
    pixels = image[mask.astype(bool)]
    if len(pixels) < 20:
        return ''
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    names, counts = np.unique(color_name(hsv), return_counts=True)
    return str(names[int(np.argmax(counts))])


# ---------------------------------------------------------------------- 3D placement
def transform_points(translation, rotation, points):
    """Points (N, 3) of a child frame in its parent: translation (x, y, z), rotation the unit
    quaternion (x, y, z, w) of a geometry_msgs/Transform (parent <- child)."""
    qx, qy, qz, qw = rotation
    matrix = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
    return np.asarray(points, dtype=float) @ matrix.T + np.asarray(translation, dtype=float)


def place_keypoints(kp, intrinsics, threshold, depth=None, window=2):
    """The keypoints in the camera frame (optical: x right, y down, z forward): (points (17, 3),
    valid (17,), from_depth). With a depth image (metres, aligned with the image) each keypoint
    takes the median depth around it; one far from the torso (the background beside an arm) and
    one without depth take the torso's. Without depth, every keypoint is at the distance that
    the size of the torso gives (a flat body facing the camera)."""
    fx, fy, cx, cy = intrinsics
    valid = kp[:, 2] >= threshold
    points = np.zeros((len(kp), 3))
    if not valid.any():
        return points, valid, False
    distances = np.full(len(kp), np.nan)
    if depth is not None:
        height, width = depth.shape
        for i in np.flatnonzero(valid):
            u, v = int(round(kp[i, 0])), int(round(kp[i, 1]))
            if not (0 <= u < width and 0 <= v < height):
                continue
            patch = depth[max(0, v - window):v + window + 1, max(0, u - window):u + window + 1]
            values = patch[np.isfinite(patch) & (patch > 0.1)]
            if len(values):
                distances[i] = float(np.median(values))
    torso_ids = [K[n] for n in ('left_shoulder', 'right_shoulder', 'left_hip', 'right_hip')]
    torso_depths = [distances[i] for i in torso_ids if valid[i] and np.isfinite(distances[i])]
    from_depth = bool(torso_depths)
    if from_depth:
        reference = float(np.median(torso_depths))
    else:
        reference = size_distance(kp, threshold, fx, fy)
        if reference is None:
            return points, np.zeros(len(kp), dtype=bool), False
    for i in np.flatnonzero(valid):
        z = distances[i]
        if not np.isfinite(z) or abs(z - reference) > 0.8:
            z = reference
        points[i] = ((kp[i, 0] - cx) / fx * z, (kp[i, 1] - cy) / fy * z, z)
    return points, valid, from_depth


def size_distance(kp, threshold, fx, fy):
    """The distance of a person from the size of their torso and shoulders in the image (the
    nearest of the two: a part seen at an angle looks smaller, never larger)."""
    estimates = []
    t = torso(kp, threshold)
    if t is not None:
        estimates.append(fy * TORSO_LENGTH_M / t[2])
    left, right = _point(kp, 'left_shoulder', threshold), _point(kp, 'right_shoulder', threshold)
    if left is not None and right is not None:
        width = float(np.linalg.norm(left - right))
        if width > 1.0:
            estimates.append(fx * SHOULDER_WIDTH_M / width)
    return min(estimates) if estimates else None


# ---------------------------------------------------------------------- over time
class Debounced:
    """A value that changes only after the new one has been seen for hold_s."""

    def __init__(self, hold_s, initial):
        self.hold_s = hold_s
        self.value = initial
        self.candidate, self.since = initial, None

    def update(self, t, value):
        """Returns True if the value changed now."""
        if value == self.value:
            self.candidate, self.since = value, None
            return False
        if value != self.candidate or self.since is None:
            self.candidate, self.since = value, t
        if t - self.since >= self.hold_s:
            self.value, self.since = value, None
            return True
        return False


class Gesture:
    """A gesture reported after hold_s and kept release_s after it is no longer seen."""

    def __init__(self, hold_s, release_s):
        self.hold_s, self.release_s = hold_s, release_s
        self.active = False
        self.first_seen, self.last_seen = None, None

    def update(self, t, seen):
        """Returns True if the gesture started now."""
        if seen:
            if self.first_seen is None:
                self.first_seen = t
            self.last_seen = t
            if not self.active and t - self.first_seen >= self.hold_s:
                self.active = True
                return True
            return False
        if self.last_seen is None or t - self.last_seen > self.release_s:
            self.active = False
            self.first_seen = None
        return False


class Waving:
    """A hand over the elbow moving sideways, back and forth."""

    def __init__(self, p):
        self.p = p
        self.history = deque()     # (t, x of the wrist from the elbow, in torso lengths)

    def update(self, t, kp):
        p = self.p
        while self.history and t - self.history[0][0] > p.waving_window_s:
            self.history.popleft()
        tr = torso(kp, p.keypoint_threshold)
        best = None
        if tr is not None:
            for side in SIDES:
                wrist = _point(kp, f'{side}_wrist', p.keypoint_threshold)
                elbow = _point(kp, f'{side}_elbow', p.keypoint_threshold)
                shoulder = _point(kp, f'{side}_shoulder', p.keypoint_threshold)
                if wrist is None or elbow is None or shoulder is None:
                    continue
                # The hand up: over the elbow and about the height of the shoulder or higher
                if wrist[1] < elbow[1] and wrist[1] < shoulder[1] + 0.1 * tr[2]:
                    x = (wrist[0] - elbow[0]) / tr[2]
                    if best is None or wrist[1] < best[1]:
                        best = (x, wrist[1])
        if best is None:
            self.history.clear()
            return False
        self.history.append((t, best[0]))
        return self.reversals() >= p.waving_reversals

    def reversals(self):
        """Changes of direction of at least waving_amplitude (a zigzag with hysteresis)."""
        amplitude = self.p.waving_amplitude
        values = [x for _, x in self.history]
        if len(values) < 3:
            return 0
        count, direction = 0, 0
        low = high = extreme = values[0]
        for x in values[1:]:
            if direction == 0:
                low, high = min(low, x), max(high, x)
                if x - low >= amplitude:
                    direction, extreme = 1, x
                elif high - x >= amplitude:
                    direction, extreme = -1, x
            elif direction > 0:
                if x > extreme:
                    extreme = x
                elif extreme - x >= amplitude:
                    count, direction, extreme = count + 1, -1, x
            else:
                if x < extreme:
                    extreme = x
                elif x - extreme >= amplitude:
                    count, direction, extreme = count + 1, 1, x
        return count


@dataclass
class PersonState:
    """What is known of a tracked person, over time."""

    id: str
    posture: Debounced
    gestures: dict
    waving: Waving
    last_seen: float = 0.0
    lying_since: object = None
    not_lying_since: object = None
    fallen: bool = False
    active_gestures: list = field(default_factory=list)


@dataclass
class Observation:
    """A person in one image: the input of BodyAnalyzer.update."""

    id: str
    keypoints: np.ndarray              # (17, 3): x, y (pixels), score
    box: tuple = None                  # x0, y0, x1, y1 (pixels)
    points: np.ndarray = None          # (17, 3) in the fixed frame (z up), or None
    valid: np.ndarray = None           # (17,) which points are placed
    from_depth: bool = False           # the points come from a depth image
    clothing_color: str = ''


@dataclass
class BodyReport:
    """What BodyAnalyzer.update says of a person in one image."""

    id: str
    posture: str
    fallen: bool
    gestures: list
    facing_camera: bool
    position: object = None            # the torso, fixed frame
    head: object = None
    pointing_direction: object = None
    pointed_location: object = None


class BodyAnalyzer:
    """Posture, fall and gestures of every tracked person, over the images."""

    def __init__(self, params=None):
        self.p = params or BodyParameters()
        self.people = {}

    def state(self, person_id):
        state = self.people.get(person_id)
        if state is None:
            p = self.p
            state = PersonState(
                person_id, Debounced(p.posture_hold_s, UNKNOWN),
                {g: Gesture(p.gesture_hold_s, p.gesture_release_s) for g in GESTURES},
                Waving(p))
            self.people[person_id] = state
        return state

    def update(self, t, observations):
        """observations: [Observation] of the image at time t (s). Returns ([BodyReport],
        [(event, person id)])."""
        p = self.p
        reports, events = [], []
        for o in observations:
            state = self.state(o.id)
            state.last_seen = t
            kp = o.keypoints
            has_3d = o.points is not None and o.valid is not None
            two_d = posture_2d(kp, p, o.box)
            three_d = posture_3d(o.points, o.valid, p) if has_3d else None
            if two_d == LYING or (three_d == LYING and o.from_depth):
                posture = LYING
            elif two_d != UNKNOWN:
                posture = two_d          # the knees are seen: the image decides
            else:
                # No knees: the height of the hips decides (a torso seen from above is short in
                # depth, so lying needs a depth image)
                posture = three_d if three_d not in (None, LYING) else UNKNOWN
            if state.posture.update(t, posture):
                events.append((state.posture.value, o.id))
            position = torso_point(o.points, o.valid) if has_3d else None

            # Fall: lying for fall_confirm_s, on the floor (when the torso is placed in 3D)
            if state.posture.value == LYING:
                state.not_lying_since = None
                if state.lying_since is None:
                    state.lying_since = t
                on_floor = position is None or position[2] - p.floor_z < p.fall_max_height_m
                if not state.fallen and on_floor and t - state.lying_since >= p.fall_confirm_s:
                    state.fallen = True
                    events.append((FALL, o.id))
            else:
                state.lying_since = None
                if state.fallen:
                    if state.not_lying_since is None:
                        state.not_lying_since = t
                    if t - state.not_lying_since >= p.recover_s:
                        state.fallen = False
                        state.not_lying_since = None
                        events.append((RECOVERED, o.id))

            # The gestures of a person lying are not read: the rules expect an upright body
            upright = two_d != LYING and posture != LYING
            raised = raised_hands(kp, p) if upright else set()
            pointing = pointing_arms(kp, p, o.points, o.valid, o.from_depth and has_3d) \
                if upright else set()
            seen = {
                'hand_raised_left': 'left' in raised and 'right' not in raised,
                'hand_raised_right': 'right' in raised and 'left' not in raised,
                'both_hands_raised': raised == {'left', 'right'},
                'waving': state.waving.update(t, kp) and upright,
                'pointing_left': 'left' in pointing,
                'pointing_right': 'right' in pointing,
            }
            for name, gesture in state.gestures.items():
                if gesture.update(t, seen[name]):
                    events.append((name, o.id))
            state.active_gestures = [g for g in GESTURES if state.gestures[g].active]

            report = BodyReport(o.id, state.posture.value, state.fallen,
                                list(state.active_gestures), facing_camera(kp, p), position)
            if has_3d:
                report.head = head_point(o.points, o.valid)
                for side in SIDES:
                    if f'pointing_{side}' not in state.active_gestures:
                        continue
                    s, w = K[f'{side}_shoulder'], K[f'{side}_wrist']
                    if o.valid[s] and o.valid[w]:
                        report.pointing_direction, report.pointed_location = \
                            pointed_location(o.points[s], o.points[w], p)
                        break
            reports.append(report)
        for person_id in [i for i, s in self.people.items() if t - s.last_seen > p.forget_s]:
            del self.people[person_id]
        return reports, events


def matches(gesture_query, gestures):
    """The gesture of gestures that satisfies a client's query ('' for any), or ''."""
    if not gestures:
        return ''
    if not gesture_query:
        return gestures[0]
    wanted = GESTURE_QUERIES.get(gesture_query, (gesture_query,))
    return next((g for g in gestures if g in wanted), '')
