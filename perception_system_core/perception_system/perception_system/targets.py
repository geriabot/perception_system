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

"""Where the target of a target frame request is, in the camera frame, from the 3D detections of
its detection request. No ROS graph here: the node places the point in the fixed frame and
publishes it.

SingleTarget follows one tracked detection (class and id). GroupTarget follows the midpoint of N
detections of a class: it keeps the ids of the first N seen, and replaces an id missing for
MAX_MISSING_ITERATIONS messages by another detection of the class. Both reproduce the TF
publishers of the attention system (pub_single_detection_tf_node and
pub_n_detections_same_class_tf_node), message by message. EachTarget gives a point per tracked
detection of a class.
"""

from dataclasses import dataclass
import math

# Half the width of a person's body (m), added to the half width of a group
BODY_HALF_WIDTH = 0.3
# Messages without a member of the group before it is replaced
MAX_MISSING_ITERATIONS = 5


@dataclass
class CameraPoint:
    """A point in the frame of the image (the header of the detections), at its time."""

    frame_id: str
    stamp: object      # builtin_interfaces/Time
    x: float
    y: float
    z: float


def center(detection):
    position = detection.bbox.center.position
    return position.x, position.y, position.z


class SingleTarget:
    """One detection, by class and tracker id."""

    def __init__(self, class_name, detection_id):
        self.class_name = class_name
        self.detection_id = detection_id
        self.pending = None      # the point not yet published
        self.seen = False        # the detection was in the last message

    def update(self, detections_msg, log=None):
        """A Detection3DArray of the detection request. Returns True if the target is in it."""
        for detection in detections_msg.detections:
            if not detection.results:
                continue
            if detection.id == self.detection_id and \
                    detection.results[0].hypothesis.class_id == self.class_name:
                x, y, z = center(detection)
                self.pending = CameraPoint(detection.header.frame_id, detection.header.stamp,
                                           x, y, z)
                self.seen = True
                return True
        self.pending = None
        self.seen = False
        if log is not None:
            log(f'Detection id={self.detection_id} class={self.class_name} not found in the '
                'current message')
        return False

    def take(self):
        """The point to publish, once."""
        point, self.pending = self.pending, None
        return point


class GroupTarget:
    """The midpoint of n detections of a class (n <= 0: of all the detections of the class)."""

    def __init__(self, class_name, n_detections):
        self.class_name = class_name
        self.n_detections = n_detections
        self.tracked_ids = []
        self.missing = {}           # id -> consecutive messages without it
        self.pending = None
        self.seen = False
        self.half_width = None      # of the last update, m

    def reset(self):
        self.pending = None
        self.seen = False
        self.tracked_ids = []
        self.missing = {}

    def update(self, detections_msg, log=None):
        """A Detection3DArray of the detection request. Returns True if the group is in it."""
        valid, by_id = [], {}
        for detection in detections_msg.detections:
            if not detection.results:
                continue
            if detection.results[0].hypothesis.class_id != self.class_name:
                continue
            if not detection.id:     # unconfirmed tracks have no id yet
                continue
            valid.append(detection)
            by_id[detection.id] = detection
        self.pending = None
        self.half_width = None

        if self.n_detections <= 0:
            self.tracked_ids = [d.id for d in valid]
            self.missing = {}
            self.seen = bool(valid) and self._set_point(valid)
            return self.seen

        if not self.tracked_ids:
            if len(valid) < self.n_detections:
                self.seen = False
                if log is not None:
                    log(f'Waiting for {self.n_detections} detections of class {self.class_name}. '
                        f'Current valid detections: {len(valid)}')
                return False
            self.tracked_ids = [d.id for d in valid[:self.n_detections]]
            self.missing = {i: 0 for i in self.tracked_ids}

        tracked = []
        all_found = True
        for tracked_id in self.tracked_ids:
            if tracked_id in by_id:
                tracked.append(by_id[tracked_id])
                self.missing[tracked_id] = 0
            else:
                all_found = False
                tracked.append(None)
                self.missing[tracked_id] = self.missing.get(tracked_id, 0) + 1
        if all_found and self._set_point(tracked):
            self.seen = True
            return True

        tracked_set = set(self.tracked_ids)
        for i, previous in enumerate(list(self.tracked_ids)):
            if previous not in self.missing or self.missing[previous] < MAX_MISSING_ITERATIONS:
                continue
            replacement = next((d for d in valid if d.id not in tracked_set), None)
            if replacement is None:
                if log is not None:
                    log(f'No replacement candidate available for missing id={previous}')
                continue
            tracked_set.discard(previous)
            self.tracked_ids[i] = replacement.id
            tracked_set.add(replacement.id)
            del self.missing[previous]
            self.missing[replacement.id] = 0

        tracked = [by_id.get(i) for i in self.tracked_ids]
        if None not in tracked and self._set_point(tracked):
            self.seen = True
            return True
        self.seen = False
        return False

    def _set_point(self, detections):
        if not detections:
            return False
        count = float(len(detections))
        sx = sum(center(d)[0] for d in detections)
        sy = sum(center(d)[1] for d in detections)
        sz = sum(center(d)[2] for d in detections)
        first = detections[0]
        cx, cz = sx / count, sz / count
        # Half width across the line of sight (optical frame: x right, z forward): the largest
        # distance of a member to the line from the camera to the midpoint, plus half a body
        norm = math.hypot(cx, cz)
        if norm > 1e-6:
            px, pz = cz / norm, -cx / norm
            half_width = 0.0
            for detection in detections:
                x, _, z = center(detection)
                half_width = max(half_width, abs((x - cx) * px + (z - cz) * pz))
            self.half_width = half_width + BODY_HALF_WIDTH
        self.pending = CameraPoint(first.header.frame_id, first.header.stamp,
                                   sx / count, sy / count, sz / count)
        return True

    def take(self):
        point, self.pending = self.pending, None
        return point


class EachTarget:
    """Every tracked detection of a class (every class if class_name is empty), each with its own
    frame: take() returns [(id, CameraPoint)]."""

    def __init__(self, class_name):
        self.class_name = class_name
        self.pending = None
        self.seen = False
        self.tracked_ids = []

    def update(self, detections_msg, log=None):
        points = []
        for detection in detections_msg.detections:
            if not detection.results or not detection.id:
                continue
            if self.class_name and detection.results[0].hypothesis.class_id != self.class_name:
                continue
            x, y, z = center(detection)
            points.append((detection.id, CameraPoint(detection.header.frame_id,
                                                     detection.header.stamp, x, y, z)))
        self.tracked_ids = [i for i, _ in points]
        self.pending = points or None
        self.seen = bool(points)
        return self.seen

    def take(self):
        points, self.pending = self.pending, None
        return points
