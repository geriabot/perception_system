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

"""The 3D placement of the detections of every request, with the estimators of
detection_3d_estimators: each detection on the ray of the center of its box, at the distance
that the estimator gives. The same computation as its node (detection_3d_estimator), done in the
perception system for each request, on the image that was processed."""

import os

from ament_index_python.packages import get_package_share_directory
from detection_3d_estimators.estimators import (BoxSizeEstimator, DepthAnythingEstimator,
                                                DepthImageEstimator, RayEstimator)
from detection_3d_estimators.geometry import Box, Intrinsics, point_at, to_body
from detection_3d_estimators.node import depth_to_metres
from detection_3d_estimators.object_sizes import ObjectSizes
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

ESTIMATORS = ('none', 'depth_image', 'ray', 'box_size', 'depth_anything')

# The parameters of the projection (prefix projection.) and their defaults
DEFAULTS = {
    'estimator': 'none',
    'camera_info_topic': '/camera_info',
    'depth_image_topic': '/depth_image',
    'camera_intrinsics': [0.0],
    'camera_frame_is_optical': True,
    'camera_frame_id': '',
    'depth_window_px': 5,
    'depth_region': 0.6,
    'depth_percentile': 10.0,
    'ray_distance_m': 1.0,
    'object_sizes_file': '',
    'sizes': [''],
    'classes': [''],
    'cut_margin': 0.025,
    'unknown_distance_m': 1.0,
    'depth_anything_model': 'depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf',
    'depth_anything_device': 'cuda',
    'marker_scale': 0.05,
    'marker_color': [0.0, 1.0, 0.0, 1.0],
}
MARKER_NAMESPACE = 'detection_centroids'


def make_estimator(p):
    """The estimator of the parameters p (DEFAULTS), and a description for the log."""
    name = p['estimator']
    if name == 'depth_image':
        return DepthImageEstimator(p['depth_window_px'], p['depth_region'],
                                   p['depth_percentile']), (
            f'{p["depth_image_topic"]}, percentile {p["depth_percentile"]:g} of the central '
            f'{p["depth_region"]:g} of each box')
    if name == 'ray':
        return RayEstimator(p['ray_distance_m']), f'every detection at {p["ray_distance_m"]} m'
    if name == 'box_size':
        path = p['object_sizes_file'] or os.path.join(
            get_package_share_directory('detection_3d_estimators'), 'config', 'object_sizes.yaml')
        sizes = ObjectSizes.from_file(path)
        sizes.override([s for s in p['sizes'] if s])
        sizes.restrict([c for c in p['classes'] if c])
        return BoxSizeEstimator(sizes, p['unknown_distance_m'], p['cut_margin']), (
            f'{len(sizes.sizes)} known sizes ({path}); the rest at {p["unknown_distance_m"]} m')
    if name == 'depth_anything':
        estimator = DepthAnythingEstimator(p['depth_anything_model'], p['depth_anything_device'],
                                           p['depth_region'], p['depth_percentile'],
                                           p['unknown_distance_m'])
        return estimator, f'{p["depth_anything_model"]} on {estimator.device}'
    raise ValueError(f'Unknown estimator "{name}": one of {", ".join(ESTIMATORS)}')


class Projector:
    """Places 2D detections in 3D. Needs the intrinsics (set_intrinsics) and, for depth_image,
    the latest depth image (set_depth)."""

    def __init__(self, params):
        self.p = dict(DEFAULTS, **params)
        self.estimator, self.description = make_estimator(self.p)
        self.optical = self.p['camera_frame_is_optical']
        self.intrinsics = None
        self.last_depth = None
        intrinsics = [float(v) for v in self.p['camera_intrinsics']]
        if len(intrinsics) == 4 and intrinsics[0] > 0.0:
            self.intrinsics = Intrinsics(*intrinsics)

    @property
    def needs_depth(self):
        return self.estimator.needs_depth

    def set_camera_info(self, msg):
        """Returns True if the intrinsics were set now."""
        if self.intrinsics is not None:
            return False
        self.intrinsics = Intrinsics(msg.k[0], msg.k[4], msg.k[2], msg.k[5], msg.width,
                                     msg.height)
        return True

    def set_depth(self, msg):
        self.last_depth = msg

    def ready(self):
        return self.intrinsics is not None

    def project(self, detections_2d, image=None):
        """A Detection2DArray in 3D: (Detection3DArray, the index in detections_2d.detections of
        each detection placed). image: the RGB image of the detections (depth_anything). Returns
        (None, reason) if the detections cannot be placed (no intrinsics yet)."""
        if self.intrinsics is None:
            return None, 'no camera calibration yet'
        header = detections_2d.header
        if not header.frame_id and self.p['camera_frame_id']:
            header.frame_id = self.p['camera_frame_id']
        indices = [i for i, d in enumerate(detections_2d.detections) if d.results]
        detections = [detections_2d.detections[i] for i in indices]
        boxes = [Box(d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.size_x,
                     d.bbox.size_y) for d in detections]
        labels = [d.results[0].hypothesis.class_id for d in detections]
        depth = None
        if self.estimator.needs_depth and detections and self.last_depth is not None:
            depth = depth_to_metres(self.last_depth)
        estimates = self.estimator.distances(self.intrinsics, boxes, labels, image, depth) \
            if boxes else []
        out = Detection3DArray(header=header)
        placed = []
        for index, det2d, box, (distance, _) in zip(indices, detections, boxes, estimates):
            if distance is None:   # no estimate (no depth there): left out
                continue
            placed.append(index)
            optical = point_at(self.intrinsics, box.u, box.v, distance)
            x, y, z = optical if self.optical else to_body(optical)
            det3d = Detection3D(header=header, id=det2d.id)
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = det2d.results[0].hypothesis.class_id
            hypothesis.hypothesis.score = det2d.results[0].hypothesis.score
            hypothesis.pose.pose.position.x = x
            hypothesis.pose.pose.position.y = y
            hypothesis.pose.pose.position.z = z
            det3d.results.append(hypothesis)
            det3d.bbox.center.position.x = x
            det3d.bbox.center.position.y = y
            det3d.bbox.center.position.z = z
            # The size of the box at the depth of its center (along the optical axis)
            det3d.bbox.size.x = box.width / self.intrinsics.fx * optical[2]
            det3d.bbox.size.y = box.height / self.intrinsics.fy * optical[2]
            det3d.bbox.size.z = 0.01
            out.detections.append(det3d)
        return out, placed

    def markers(self, detections_3d, previous_count):
        """The RViz spheres of the detections, deleting those of a previous message beyond them."""
        header = detections_3d.header
        markers = MarkerArray()
        scale = self.p['marker_scale']
        r, g, b, a = (list(self.p['marker_color']) + [0.0, 1.0, 0.0, 1.0])[:4]
        for i, detection in enumerate(detections_3d.detections):
            marker = Marker(header=header, ns=MARKER_NAMESPACE, id=i, type=Marker.SPHERE,
                            action=Marker.ADD)
            position = detection.bbox.center.position
            marker.pose.position.x, marker.pose.position.y = position.x, position.y
            marker.pose.position.z = position.z
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = scale
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = r, g, b, a
            markers.markers.append(marker)
        for old in range(len(detections_3d.detections), previous_count):
            markers.markers.append(Marker(header=header, ns=MARKER_NAMESPACE, id=old,
                                          action=Marker.DELETE))
        return markers
