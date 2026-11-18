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

"""Names, namespaces, outputs and conflicts of the requests; the 3D placement."""

import math

import pytest
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from perception_system import projection, requests as rq
from perception_system.engine import RequestConfig
from perception_system_interfaces.srv import StartDetection

DEFAULTS = {'model': 'omdet', 'model_options': {'threshold': 0.3}, 'tracker': 'strongsort'}


def test_names():
    rq.check_name('attention_1')
    for bad in ('', '1a', 'a-b', 'a/b', 'a b'):
        with pytest.raises(rq.RequestError):
            rq.check_name(bad)


def test_namespaces():
    base = '/perception_system/requests'
    assert rq.namespace_of('', 'attention', base) == '/perception_system/requests/attention'
    assert rq.namespace_of('/', 'attention', base) == ''
    assert rq.namespace_of('robot/eyes/', 'x', base) == '/robot/eyes'
    with pytest.raises(rq.RequestError):
        rq.namespace_of('/bad name', 'x', base)


def test_outputs_and_topics():
    assert rq.outputs_of(0) == rq.OUTPUT_DEFAULT
    with pytest.raises(rq.RequestError):
        rq.outputs_of(16)
    assert rq.topics_of('', rq.OUTPUT_DEFAULT) == {
        1: '/detections_2d', 2: '/detections_3d', 4: '/detections_image'}
    assert rq.topics_of('/a', rq.OUTPUT_DETECTIONS_2D | rq.OUTPUT_DEBUG_IMAGE) == {
        1: '/a/detections_2d', 8: '/a/debug_image'}
    # The values are those of the service
    request = StartDetection.Request
    assert (request.OUTPUT_DETECTIONS_2D, request.OUTPUT_DETECTIONS_3D,
            request.OUTPUT_DETECTIONS_IMAGE, request.OUTPUT_DEBUG_IMAGE) == (1, 2, 4, 8)


def test_engine_config_defaults_and_overrides():
    request = StartDetection.Request(name='a', class_name='person in red', category='person')
    config = rq.engine_config(request, DEFAULTS)
    assert (config.model, config.options, config.tracker) == ('omdet', {'threshold': 0.3},
                                                              'strongsort')
    request.model, request.model_options, request.tracker = 'yolo', '{imgsz: 320}', 'none'
    config = rq.engine_config(request, DEFAULTS)
    assert (config.model, config.options, config.tracker) == ('yolo', {'imgsz': 320}, 'none')
    for field, value in (('tracker', 'sort'), ('model_options', '[1, 2]'),
                         ('model_options', '{a: [')):
        bad = StartDetection.Request(name='a', **{field: value})
        with pytest.raises(rq.RequestError):
            rq.engine_config(bad, DEFAULTS)


def test_conflicts():
    open_requests = {
        'a': rq.DetectionRequest('a', RequestConfig('omdet', {}, 'none'), 7, ''),
        'f': rq.FrameRequest('f', 'a', 'target', 'odom', 0, 'person', '1', 0, ''),
        's': rq.SkillRequest('s', 'lip_activity'),
    }
    rq.check_conflicts(open_requests, 'a', rq.DetectionRequest, namespace='')   # itself
    with pytest.raises(rq.RequestError, match='publishes on /'):
        rq.check_conflicts(open_requests, 'b', rq.DetectionRequest, namespace='')
    with pytest.raises(rq.RequestError, match='another kind'):
        rq.check_conflicts(open_requests, 's', rq.DetectionRequest, namespace='/x')
    with pytest.raises(rq.RequestError, match='frame target'):
        rq.check_conflicts(open_requests, 'g', rq.FrameRequest, frame_id='target')


def detections(*boxes, frame='camera_optical'):
    msg = Detection2DArray()
    msg.header.frame_id = frame
    for i, (u, v, w, h, label) in enumerate(boxes):
        d = Detection2D(header=msg.header, id=str(i + 1))
        d.bbox.center.position.x, d.bbox.center.position.y = u, v
        d.bbox.size_x, d.bbox.size_y = w, h
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id, hypothesis.hypothesis.score = label, 0.9
        d.results.append(hypothesis)
        msg.detections.append(d)
    return msg


def camera_info(fx=500.0, cx=320.0, cy=240.0):
    info = CameraInfo(width=640, height=480)
    info.k = [fx, 0.0, cx, 0.0, fx, cy, 0.0, 0.0, 1.0]
    return info


def test_projection_ray_in_the_body_frame():
    projector = projection.Projector({'estimator': 'ray', 'camera_frame_is_optical': False})
    out, reason = projector.project(detections((320, 240, 50, 100, 'person')))
    assert out is None and reason
    projector.set_camera_info(camera_info())
    out, _ = projector.project(detections((320, 240, 50, 100, 'person'),
                                          (820, 240, 50, 100, 'person')))
    first, second = out.detections
    p = first.bbox.center.position
    assert (p.x, p.y, p.z) == pytest.approx((1.0, 0.0, 0.0))       # x forward
    p = second.bbox.center.position
    # 500 px right of the center at fx 500: 45 degrees to the right (y left)
    assert (p.x, p.y) == pytest.approx((math.sqrt(0.5), -math.sqrt(0.5)))
    assert first.id == '1' and first.results[0].hypothesis.class_id == 'person'


def test_projection_depth_image():
    projector = projection.Projector({'estimator': 'depth_image',
                                      'camera_intrinsics': [500.0, 500.0, 320.0, 240.0]})
    depth = Image(width=640, height=480, encoding='32FC1', step=640 * 4)
    import numpy as np
    values = np.full((480, 640), 3.0, dtype=np.float32)
    values[200:280, 300:340] = 2.0          # the target, in front of a wall at 3 m
    depth.data = values.tobytes()
    projector.set_depth(depth)
    out, placed = projector.project(detections((320, 240, 40, 80, 'box'),
                                               (100, 100, 20, 20, 'x')))
    target, wall = out.detections
    assert placed == [0, 1]
    assert target.bbox.center.position.z == pytest.approx(2.0)      # optical: z forward
    assert wall.bbox.center.position.z == pytest.approx(3.0)
    # No depth image: nothing placed (left out)
    projector.last_depth = None
    out, placed = projector.project(detections((320, 240, 40, 80, 'box')))
    assert out.detections == [] and placed == []
    markers = projector.markers(out, previous_count=2)
    assert [m.action for m in markers.markers] == [2, 2]            # the old spheres deleted
