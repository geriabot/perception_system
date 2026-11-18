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

"""The rules of the body detector on hand-made skeletons (no model): posture, fall, gestures,
pointing, facing, the color of the clothes and the 3D placement."""

import numpy as np
import pytest

from perception_body import body as b

P = b.BodyParameters()
# A person facing the camera, standing, 100 px of torso (their left is on the right of the image)
STANDING = {
    'nose': (100, 40), 'left_eye': (106, 35), 'right_eye': (94, 35), 'left_ear': (112, 38),
    'right_ear': (88, 38), 'left_shoulder': (125, 70), 'right_shoulder': (75, 70),
    'left_elbow': (130, 110), 'right_elbow': (70, 110), 'left_wrist': (132, 150),
    'right_wrist': (68, 150), 'left_hip': (115, 170), 'right_hip': (85, 170),
    'left_knee': (115, 230), 'right_knee': (85, 230), 'left_ankle': (115, 290),
    'right_ankle': (85, 290)}
BOX = (60, 20, 140, 300)


def skeleton(**changes):
    points = dict(STANDING, **changes)
    return np.array([[*points[name], 0.9] for name in b.KEYPOINT_NAMES], dtype=float)


def rotated(kp, degrees, center=(100, 170)):
    angle = np.radians(degrees)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    out = kp.copy()
    out[:, :2] = (kp[:, :2] - center) @ rotation.T + center
    return out


SITTING = dict(left_knee=(170, 175), right_knee=(140, 175), left_ankle=(170, 235),
               right_ankle=(140, 235))
LEFT_RAISED = dict(left_elbow=(135, 40), left_wrist=(140, 0))
RIGHT_RAISED = dict(right_elbow=(65, 40), right_wrist=(60, 0))
RIGHT_POINTING = dict(right_elbow=(40, 72), right_wrist=(5, 74))


def test_posture():
    assert b.posture_2d(skeleton(), P, BOX) == b.STANDING
    assert b.posture_2d(skeleton(**SITTING), P, BOX) == b.SITTING
    assert b.posture_2d(rotated(skeleton(), 90), P) == b.LYING
    no_knees = skeleton()
    no_knees[b.K['left_knee'], 2] = no_knees[b.K['right_knee'], 2] = 0.1
    assert b.posture_2d(no_knees, P, BOX) == b.UNKNOWN


def test_posture_3d_hip_height():
    points, valid = np.zeros((17, 3)), np.zeros(17, dtype=bool)
    for name, z in (('left_shoulder', 1.4), ('right_shoulder', 1.4), ('left_hip', 0.95),
                    ('right_hip', 0.95)):
        points[b.K[name]] = (2.0, 0.1 if 'left' in name else -0.1, z)
        valid[b.K[name]] = True
    assert b.posture_3d(points, valid, P) == b.STANDING
    points[:, 2] -= 0.45
    assert b.posture_3d(points, valid, P) == b.SITTING


def test_hands_and_pointing():
    assert b.raised_hands(skeleton(), P) == set()
    assert b.raised_hands(skeleton(**LEFT_RAISED), P) == {'left'}
    assert b.raised_hands(skeleton(**LEFT_RAISED, **RIGHT_RAISED), P) == {'left', 'right'}
    assert b.pointing_arms(skeleton(), P) == set()
    assert b.pointing_arms(skeleton(**RIGHT_POINTING), P) == {'right'}
    # A raised hand is not pointing, and a bent arm does not point
    assert b.pointing_arms(skeleton(**RIGHT_RAISED), P) == set()
    assert b.pointing_arms(skeleton(right_elbow=(40, 72), right_wrist=(45, 30)), P) == set()


def test_facing():
    assert b.facing_camera(skeleton(), P)
    assert not b.facing_camera(skeleton(nose=(95, 40)), P)      # the nose at an eye
    turned = skeleton()
    turned[b.K['left_eye'], 2] = 0.1
    assert not b.facing_camera(turned, P)


def test_pointed_location():
    direction, point = b.pointed_location((0.0, 0.0, 1.4), (0.5, 0.0, 1.1), P)
    assert np.allclose(direction, np.array([0.5, 0.0, -0.3]) / np.hypot(0.5, 0.3))
    assert point == pytest.approx([0.5 + 1.1 * 0.5 / 0.3, 0.0, 0.0])
    # Level: it does not meet the floor
    assert b.pointed_location((0.0, 0.0, 1.4), (0.5, 0.0, 1.4), P)[1] is None


def test_clothing_color():
    image = np.full((320, 200, 3), 255, dtype=np.uint8)
    image[60:180, 70:130] = (200, 20, 20)
    assert b.clothing_color(image, skeleton(), P) == 'red'
    image[60:180, 70:130] = (30, 60, 200)
    assert b.clothing_color(image, skeleton(), P) == 'blue'
    image[60:180, 70:130] = (10, 10, 10)
    assert b.clothing_color(image, skeleton(), P) == 'black'


def test_place_keypoints():
    intrinsics = (500.0, 500.0, 100.0, 160.0)
    points, valid, from_depth = b.place_keypoints(skeleton(), intrinsics, 0.5)
    assert valid.all() and not from_depth
    # 100 px of torso with fy 500: 2.5 m (the shoulders give 500 * 0.36 / 50 = 3.6 m: farther)
    assert points[:, 2] == pytest.approx(np.full(17, 2.5))
    depth = np.full((320, 200), 3.0)
    depth[0:20, :] = 9.0      # the background above the head: replaced by the torso's depth
    points, valid, from_depth = b.place_keypoints(skeleton(**LEFT_RAISED), intrinsics, 0.5,
                                                  depth)
    assert from_depth and points[:, 2] == pytest.approx(np.full(17, 3.0))
    assert points[b.K['nose']] == pytest.approx([0.0, (40 - 160) / 500 * 3.0, 3.0])


def run(analyzer, frames, t0=0.0, rate=15.0):
    """frames: [(seconds, keypoints or callable(t) -> keypoints)]. Returns [(t, event)]."""
    events, t = [], t0
    for seconds, kp in frames:
        end = t + seconds
        while t < end:
            current = kp(t) if callable(kp) else kp
            _, new = analyzer.update(t, [b.Observation('1', current, BOX)])
            events += [(round(t, 2), e) for e, _ in new]
            t += 1.0 / rate
    return events, t


def test_gesture_hold_and_events():
    analyzer = b.BodyAnalyzer(P)
    events, t = run(analyzer, [(1.0, skeleton()), (0.2, skeleton(**LEFT_RAISED)),
                               (1.0, skeleton())])
    assert [e for _, e in events] == ['standing']          # 0.2 s is not a gesture
    events, _ = run(analyzer, [(1.0, skeleton(**LEFT_RAISED))], t)
    assert [e for _, e in events] == ['hand_raised_left']
    assert analyzer.people['1'].active_gestures == ['hand_raised_left']


def test_waving():
    def wave(t):
        x = 140 + 25 * np.sin(2 * np.pi * 2.0 * t)      # 2 Hz, 50 px wide (0.5 torso)
        return skeleton(left_elbow=(135, 60), left_wrist=(x, 20))
    analyzer = b.BodyAnalyzer(P)
    events, _ = run(analyzer, [(2.0, wave)])
    assert 'waving' in [e for _, e in events]
    # The hand still and raised: raised, not waving
    analyzer = b.BodyAnalyzer(P)
    events, _ = run(analyzer, [(2.0, skeleton(left_elbow=(135, 60), left_wrist=(140, 20)))])
    assert [e for _, e in events if e != 'standing'] == ['hand_raised_left']


def test_fall_and_recovery():
    analyzer = b.BodyAnalyzer(P)
    lying = rotated(skeleton(), 90)
    events, t = run(analyzer, [(1.0, skeleton()), (2.0, lying)])
    names = [e for _, e in events]
    assert names == ['standing', 'lying', 'fall']
    times = dict((e, s) for s, e in events)
    # Lying is reported after posture_hold_s, the fall fall_confirm_s later
    assert times['fall'] - times['lying'] == pytest.approx(P.fall_confirm_s, abs=0.1)
    assert analyzer.people['1'].fallen
    # The gestures of a person lying are not read
    events, t = run(analyzer, [(1.0, rotated(skeleton(**LEFT_RAISED), 90))], t)
    assert events == []
    events, _ = run(analyzer, [(2.0, skeleton())], t)
    assert [e for _, e in events] == ['standing', 'recovered']


def test_lying_on_a_bed_is_not_a_fall():
    analyzer = b.BodyAnalyzer(P)
    lying = rotated(skeleton(), 90)
    points = np.tile([2.0, 0.0, 0.9], (17, 1))     # 0.9 m over the floor
    valid = np.ones(17, dtype=bool)
    events = []
    for i in range(45):
        _, new = analyzer.update(i / 15.0, [b.Observation('1', lying, None, points, valid)])
        events += [e for e, _ in new]
    assert events == ['lying']


def test_forget_and_matches():
    analyzer = b.BodyAnalyzer(P)
    analyzer.update(0.0, [b.Observation('1', skeleton(), BOX)])
    analyzer.update(5.0, [])
    assert analyzer.people == {}
    assert b.matches('', ['waving']) == 'waving'
    assert b.matches('hand_raised', ['both_hands_raised']) == 'both_hands_raised'
    assert b.matches('pointing', ['waving', 'pointing_left']) == 'pointing_left'
    assert b.matches('waving', ['pointing_left']) == ''
    assert b.matches('waving', []) == ''
