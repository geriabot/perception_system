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

"""The image of the body detector: each person's skeleton, id, posture and gestures."""

import cv2

from perception_body.body import K

BONES = [('left_shoulder', 'right_shoulder'), ('left_shoulder', 'left_elbow'),
         ('left_elbow', 'left_wrist'), ('right_shoulder', 'right_elbow'),
         ('right_elbow', 'right_wrist'), ('left_shoulder', 'left_hip'),
         ('right_shoulder', 'right_hip'), ('left_hip', 'right_hip'), ('left_hip', 'left_knee'),
         ('left_knee', 'left_ankle'), ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
         ('nose', 'left_eye'), ('nose', 'right_eye'), ('left_eye', 'left_ear'),
         ('right_eye', 'right_ear')]
GREEN, RED, YELLOW = (0, 220, 0), (230, 30, 30), (250, 210, 0)


def draw_people(image, detections, reports, threshold=0.5):
    """image: RGB. detections: PoseDetection; reports: BodyReport of each one (same order)."""
    out = image.copy()
    for d, report in zip(detections, reports):
        kp = d.keypoints
        color = RED if report.fallen else (YELLOW if report.gestures else GREEN)
        for a, b in BONES:
            if kp[K[a], 2] >= threshold and kp[K[b], 2] >= threshold:
                cv2.line(out, (int(kp[K[a], 0]), int(kp[K[a], 1])),
                         (int(kp[K[b], 0]), int(kp[K[b], 1])), color, 2, cv2.LINE_AA)
        for x, y, score in kp:
            if score >= threshold:
                cv2.circle(out, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)
        x0, y0 = int(d.box[0]), int(d.box[1])
        label = f'{d.id} {report.posture}' + (' FALLEN' if report.fallen else '')
        if report.gestures:
            label += ' ' + ','.join(report.gestures)
        cv2.putText(out, label, (x0, max(15, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                    cv2.LINE_AA)
    return out
