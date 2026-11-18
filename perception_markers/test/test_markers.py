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

"""Markers drawn at a known distance: the id, the pose and the sizes."""

import math

import numpy as np
import pytest

from perception_markers.markers import (matrix_to_quaternion, MarkerDetector, parse_sizes,
                                        quaternion_multiply)

FX, CX, CY = 500.0, 320.0, 240.0


def scene(marker_id, side_px, dictionary='DICT_4X4_50'):
    import cv2
    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)), marker_id, side_px)
    image = np.full((480, 640), 255, dtype=np.uint8)
    top, left = int(CY - side_px / 2), int(CX - side_px / 2)
    image[top:top + side_px, left:left + side_px] = marker
    return np.stack([image] * 3, axis=-1)


def test_pose_of_a_marker_facing_the_camera():
    # 200 px at fx 500: a marker of 0.4 m is 1 m away
    detector = MarkerDetector('DICT_4X4_50', size_m=0.4)
    [marker] = detector.detect(scene(7, 200), (FX, FX, CX, CY))
    assert marker.id == 7
    assert marker.position == pytest.approx((0.0, 0.0, 1.0), abs=0.01)
    assert marker.distance_m == pytest.approx(1.0, abs=0.01)
    # Its z axis out of the marker, towards the camera: (0, 0, -1) in the optical frame
    x, y, z, w = marker.orientation
    z_axis = (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))
    assert z_axis == pytest.approx((0.0, 0.0, -1.0), abs=0.02)


def test_sizes_and_dictionaries():
    detector = MarkerDetector('DICT_APRILTAG_36h11', size_m=0.4, sizes=parse_sizes(['3:0.2']))
    [marker] = detector.detect(scene(3, 200, 'DICT_APRILTAG_36h11'), (FX, FX, CX, CY))
    assert marker.id == 3 and marker.distance_m == pytest.approx(0.5, abs=0.01)
    assert detector.detect(scene(3, 200, 'DICT_APRILTAG_36h11'))[0].position is None
    assert MarkerDetector('DICT_4X4_50').detect(np.full((100, 100, 3), 255, np.uint8)) == []
    with pytest.raises(ValueError):
        MarkerDetector('DICT_NOT_A_DICTIONARY')
    with pytest.raises(ValueError):
        parse_sizes(['7'])


def test_quaternions():
    identity = matrix_to_quaternion(np.eye(3))
    assert identity == pytest.approx((0.0, 0.0, 0.0, 1.0))
    half_turn = matrix_to_quaternion(np.diag([1.0, -1.0, -1.0]))
    assert [abs(v) for v in half_turn] == pytest.approx((1.0, 0.0, 0.0, 0.0))
    quarter = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    assert quaternion_multiply(quarter, quarter) == pytest.approx((0.0, 0.0, 1.0, 0.0), abs=1e-9)
