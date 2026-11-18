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

"""The targets of the target frames, and placing a point in another frame."""

import math

from geometry_msgs.msg import Transform
import numpy as np
import pytest
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from perception_system import targets, transforms


def message(*detections, frame='camera_optical', sec=5):
    """detections: (id, class, x, y, z)."""
    msg = Detection3DArray()
    msg.header.frame_id = frame
    msg.header.stamp.sec = sec
    for det_id, class_name, x, y, z in detections:
        d = Detection3D(header=msg.header, id=det_id)
        h = ObjectHypothesisWithPose()
        h.hypothesis.class_id = class_name
        d.results.append(h)
        d.bbox.center.position.x, d.bbox.center.position.y = x, y
        d.bbox.center.position.z = z
        msg.detections.append(d)
    return msg


def test_single_target_by_class_and_id():
    target = targets.SingleTarget('person', '3')
    assert target.update(message(('3', 'chair', 0, 0, 1), ('3', 'person', 0.5, 0.1, 2.0)))
    point = target.take()
    assert (point.frame_id, point.stamp.sec, point.x, point.y, point.z) == \
        ('camera_optical', 5, 0.5, 0.1, 2.0)
    assert target.take() is None                   # published once
    assert not target.update(message(('4', 'person', 0, 0, 1)))
    assert target.pending is None and not target.seen


def test_single_target_lost_cancels_the_pending_point():
    target = targets.SingleTarget('person', '3')
    target.update(message(('3', 'person', 0, 0, 1)))
    target.update(message())
    assert target.take() is None


def test_group_waits_for_n_then_keeps_their_ids():
    group = targets.GroupTarget('person', 2)
    assert not group.update(message(('1', 'person', 0, 0, 2)))
    assert group.update(message(('1', 'person', -1, 0, 2), ('2', 'person', 1, 0, 2),
                                ('3', 'person', 3, 0, 2)))
    assert group.tracked_ids == ['1', '2']
    point = group.take()
    assert (point.x, point.y, point.z) == (0.0, 0.0, 2.0)
    # Half width across the line of sight: 1 m to each side, plus half a body
    assert group.half_width == pytest.approx(1.0 + targets.BODY_HALF_WIDTH)


def test_group_ignores_unconfirmed_and_other_classes():
    group = targets.GroupTarget('person', 2)
    assert not group.update(message(('', 'person', 0, 0, 2), ('5', 'chair', 1, 0, 2),
                                    ('6', 'person', 1, 0, 2)))


def test_group_replaces_a_member_missing_for_5_messages():
    group = targets.GroupTarget('person', 2)
    group.update(message(('1', 'person', 0, 0, 2), ('2', 'person', 1, 0, 2)))
    for _ in range(targets.MAX_MISSING_ITERATIONS - 1):
        assert not group.update(message(('1', 'person', 0, 0, 2), ('9', 'person', 4, 0, 2)))
        assert group.tracked_ids == ['1', '2']
    assert group.update(message(('1', 'person', 0, 0, 2), ('9', 'person', 4, 0, 2)))
    assert group.tracked_ids == ['1', '9']
    assert group.take().x == 2.0


def test_group_of_all_detections():
    group = targets.GroupTarget('person', 0)
    assert not group.update(message())
    assert group.update(message(('1', 'person', 0, 0, 2), ('2', 'person', 2, 0, 2),
                                ('3', 'person', 4, 0, 2)))
    assert group.tracked_ids == ['1', '2', '3'] and group.take().x == 2.0


def test_point_in_another_frame():
    t = Transform()
    t.translation.x, t.translation.y, t.translation.z = 1.0, 2.0, 0.5
    # 90 degrees about z: x -> y
    t.rotation.z, t.rotation.w = math.sin(math.pi / 4), math.cos(math.pi / 4)
    x, y, z = transforms.apply(t, (1.0, 0.0, 0.3))
    assert (x, y, z) == pytest.approx((1.0, 3.0, 0.8))


def test_rotation_matches_a_matrix():
    q = (0.1, -0.3, 0.2, 0.9)
    norm = math.sqrt(sum(c * c for c in q))
    x, y, z, w = (c / norm for c in q)
    matrix = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
              [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
              [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    v = (0.7, -1.2, 2.5)
    expected = [sum(matrix[i][j] * v[j] for j in range(3)) for i in range(3)]
    assert transforms.rotate((x, y, z, w), v) == pytest.approx(expected, abs=1e-12)


def test_each_target_one_point_per_tracked_detection():
    each = targets.EachTarget('person')
    assert each.update(message(('1', 'person', 0, 0, 2), ('', 'person', 1, 0, 2),
                               ('2', 'chair', 2, 0, 2), ('3', 'person', 3, 0, 2)))
    points = each.take()
    assert [(i, p.x) for i, p in points] == [('1', 0), ('3', 3)]
    assert each.take() is None
    every = targets.EachTarget('')
    every.update(message(('1', 'person', 0, 0, 2), ('2', 'chair', 2, 0, 2)))
    assert every.tracked_ids == ['1', '2']
    assert not every.update(message())


# ---------------------------------------------------------------------- MODE_FOLLOW
def scene(*boxes):
    """An image with colored boxes, and their 2D detections: (id, color, x0, y0, x1, y1)."""
    from perception_detectors.pipeline import Detection
    image = np.full((240, 320, 3), 255, dtype=np.uint8)
    detections = []
    for det_id, color, x0, y0, x1, y1 in boxes:
        image[y0:y1, x0:x1] = color
        detections.append(Detection(cx=(x0 + x1) / 2, cy=(y0 + y1) / 2, w=x1 - x0, h=y1 - y0,
                                     score=0.9, label='person', id=det_id))
    return image, detections


RED, BLUE = (220, 30, 30), (30, 30, 220)


def follow_target(detection_id='1'):
    from perception_system.appearance import ColorAppearance
    return targets.FollowTarget('person', detection_id, ColorAppearance(), 0.85)


def test_follow_recognizes_the_object_with_a_new_id():
    target = follow_target()
    # Seen with another person: that one is another object, even of the same appearance
    target.observe(*scene(('1', RED, 10, 10, 60, 200), ('2', BLUE, 200, 10, 250, 200)))
    assert target.others == {'2'}
    assert target.update(message(('1', 'person', 0.0, 0.0, 2.0)))
    assert target.tracked_ids == ['1']
    # Lost: a new id of another appearance is not taken; one of its appearance is
    assert target.observe(*scene(('2', BLUE, 200, 10, 250, 200),
                                 ('5', BLUE, 100, 10, 150, 200))) is None
    assert not target.update(message(('2', 'person', 1.0, 0.0, 2.0)))
    assert target.tracked_ids == []
    assert target.observe(*scene(('2', BLUE, 200, 10, 250, 200),
                                 ('7', RED, 100, 10, 150, 200))) == '7'
    assert target.detection_id == '7' and target.reidentifications == 1
    assert target.update(message(('7', 'person', 0.5, 0.0, 3.0)))
    assert target.take().z == pytest.approx(3.0)


def test_follow_never_takes_an_object_seen_with_it():
    target = follow_target()
    target.observe(*scene(('1', RED, 10, 10, 60, 200), ('2', RED, 200, 10, 250, 200)))
    # The same color, but seen at the same time: another object
    assert target.observe(*scene(('2', RED, 200, 10, 250, 200))) is None
    assert target.detection_id == '1'
