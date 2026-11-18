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

"""Fiducial markers (ArUco, AprilTag) and their 6D pose in the camera frame, with OpenCV. No ROS
here. A marker printed on a door, a charger, a shelf or an object gives the robot where it is and
how it is turned, to the centimetre, with no learned model.
"""

from dataclasses import dataclass
import math

import numpy as np

# The dictionaries of OpenCV that can be asked for by name
DICTIONARIES = ('DICT_4X4_50', 'DICT_4X4_100', 'DICT_4X4_250', 'DICT_5X5_100', 'DICT_6X6_250',
                'DICT_ARUCO_ORIGINAL', 'DICT_APRILTAG_16h5', 'DICT_APRILTAG_25h9',
                'DICT_APRILTAG_36h10', 'DICT_APRILTAG_36h11')


@dataclass
class MarkerPose:
    id: int
    corners: np.ndarray      # (4, 2) pixels: top left, top right, bottom right, bottom left
    size_m: float
    position: tuple          # x, y, z of its center in the optical frame (x right, y down, z
    orientation: tuple       # forward); quaternion x, y, z, w: its z axis out of the marker
    distance_m: float


def parse_sizes(entries):
    """'id:size' entries (e.g. ['7:0.2', '12:0.05']) as {id: size in m}."""
    sizes = {}
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        marker_id, _, size = entry.partition(':')
        try:
            sizes[int(marker_id)] = float(size)
        except ValueError as exc:
            raise ValueError(f'Invalid marker size "{entry}": id:size_in_m') from exc
    return sizes


def matrix_to_quaternion(m):
    """A rotation matrix (3, 3) as a quaternion (x, y, z, w)."""
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        return ((m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s,
                0.25 / s)
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        return (0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s,
                (m[2, 1] - m[1, 2]) / s)
    if m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        return ((m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s,
                (m[0, 2] - m[2, 0]) / s)
    s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
    return ((m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s,
            (m[1, 0] - m[0, 1]) / s)


def quaternion_multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def rotate(q, v):
    """The vector v rotated by the unit quaternion q (x, y, z, w)."""
    x, y, z, _ = quaternion_multiply(quaternion_multiply(q, (v[0], v[1], v[2], 0.0)),
                                     (-q[0], -q[1], -q[2], q[3]))
    return x, y, z


class MarkerDetector:
    """The markers of a dictionary in an image, with their pose."""

    def __init__(self, dictionary='DICT_4X4_50', size_m=0.10, sizes=None):
        import cv2
        if dictionary not in DICTIONARIES:
            raise ValueError(f'Unknown dictionary "{dictionary}": one of '
                             f'{", ".join(DICTIONARIES)}')
        self.dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, parameters)
        self.size_m = float(size_m)
        self.sizes = dict(sizes or {})

    def detect(self, image, intrinsics=None, distortion=None):
        """image: RGB or gray uint8. intrinsics: (fx, fy, cx, cy) for the pose (None: corners
        only, position and orientation None). Returns [MarkerPose]."""
        import cv2
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return []
        out = []
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            marker_id = int(marker_id)
            size = self.sizes.get(marker_id, self.size_m)
            pixels = marker_corners.reshape(4, 2).astype(np.float64)
            position = orientation = None
            distance = 0.0
            if intrinsics is not None:
                position, orientation = self.pose(pixels, size, intrinsics, distortion)
                if position is not None:
                    distance = float(np.linalg.norm(position))
            out.append(MarkerPose(marker_id, pixels, size, position, orientation, distance))
        return out

    @staticmethod
    def pose(pixels, size, intrinsics, distortion=None):
        import cv2
        fx, fy, cx, cy = intrinsics
        camera = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        half = size / 2.0
        # The corners of the marker in its own frame (z out of it), in the order of ArUco
        model = np.array([[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0],
                          [-half, -half, 0.0]])
        ok, rvec, tvec = cv2.solvePnP(
            model, pixels, camera,
            np.zeros(5) if distortion is None else np.asarray(distortion, dtype=np.float64),
            flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return None, None
        rotation, _ = cv2.Rodrigues(rvec)
        return tuple(float(v) for v in tvec.flatten()), matrix_to_quaternion(rotation)
