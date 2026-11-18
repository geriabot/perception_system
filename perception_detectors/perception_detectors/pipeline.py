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

"""From an image to the detections of the requested class, with tracker ids.

The same pipeline runs in the perception system (local engine) and in perception_server, so a
model behaves the same in the robot and in a server. One pipeline per detection request: its
class and its tracker.
"""

from dataclasses import asdict, dataclass

import numpy as np

# torch and torchvision are imported where they are used: a client of a perception server (the
# remote engine) needs only Detection, and no torch on the robot's computer

# IoU above which a detection of the class and one of its category are the same object
SAME_OBJECT_IOU = 0.5


@dataclass
class Detection:
    cx: float        # center and size of the box, in pixels
    cy: float
    w: float
    h: float
    score: float
    label: str       # always the requested class
    id: str = ''     # tracker id; empty while the track is not confirmed

    def to_dict(self):
        return asdict(self)


class DetectionPipeline:
    """The detections of the class set with set_class, in each image.

    With a category (e.g. class "person in red", category "person"), the model looks for
    both texts and every detection is published as the class: open-vocabulary detectors
    are unreliable with attributes, so every object of the category is a candidate, and the
    target is chosen among them afterwards (the MLLM, looking at the marked image).
    """

    def __init__(self, backend, tracker=None):
        self.backend = backend
        self.tracker = tracker
        self.labels = []
        self.with_category = False

    def set_class(self, class_name, category=''):
        """Returns a description of what is detected now. An empty class stops detecting."""
        class_name, category = class_name.strip(), category.strip()
        self.labels = [class_name] if class_name else []
        self.with_category = bool(
            class_name and category and category.lower() != class_name.lower())
        if self.with_category:
            self.labels.append(category)
        if not self.labels:
            return 'Detection class cleared'
        if self.with_category:
            return f'New class to detect: {class_name} (also looking for {category})'
        return f'New class to detect: {class_name}'

    def set_labels(self, labels):
        """Several independent classes (the `classes` parameter of the detector)."""
        self.labels = [label for label in labels if label]
        self.with_category = False

    def process(self, image, detect=None):
        """image: RGB uint8 (H, W, 3). Returns the detections ([] without a class).

        detect(image, labels), instead of the backend's: the perception system passes one that
        reuses the output of the model for every request with the same model and labels, so the
        model runs once per image for all of them. The output is not modified here."""
        if not self.labels:
            return []
        boxes, scores, labels = (detect or self.backend.detect)(image, self.labels)
        if self.with_category and len(labels):
            boxes, scores = merge_class_and_category(boxes, scores, labels, self.labels[1])
            labels = [self.labels[0]] * len(scores)
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            box = [round(v, 2) for v in box.tolist()]
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            detections.append(Detection(
                cx=float(x0 + x1) / 2, cy=float(y0 + y1) / 2, w=float(x1 - x0),
                h=float(y1 - y0), score=round(float(score), 2), label=label))
        if self.tracker is not None:
            self._assign_ids(image, detections)
        return detections

    def _assign_ids(self, image, detections):
        rows = np.array(
            [[d.cx - d.w / 2, d.cy - d.h / 2, d.cx + d.w / 2, d.cy + d.h / 2, d.score, 0.0]
             for d in detections], dtype=np.float32).reshape(-1, 6)
        tracks = self.tracker.update(rows, image)
        if tracks is None or len(tracks) == 0:
            return
        assigned = set()
        for track in tracks:
            x0, y0, x1, y1, track_id = track[:5]
            index = best_match(np.array([x0, y0, x1, y1], dtype=np.float32), rows, assigned,
                               self.tracker.iou_threshold)
            if index is not None:
                detections[index].id = str(int(track_id))
                assigned.add(index)


def merge_class_and_category(boxes, scores, labels, category):
    """The objects found with the class, with its category or with both, once each. The
    models already removed overlapping boxes of the same label; here only the duplicates
    across the two labels go: a box of the class over a box of the category (IoU >
    SAME_OBJECT_IOU) is the same object, kept once with the best score. (Suppressing across
    all the boxes instead would also merge different people standing close together.)"""
    import torch
    import torchvision
    is_category = torch.tensor([label == category for label in labels], dtype=torch.bool)
    category_idx = torch.nonzero(is_category).flatten()
    class_idx = torch.nonzero(~is_category).flatten()
    boxes, scores = boxes.float().cpu(), scores.float().cpu()
    keep_boxes = [boxes[i] for i in category_idx.tolist()]
    keep_scores = [scores[i] for i in category_idx.tolist()]
    if len(class_idx):
        overlap = (torchvision.ops.box_iou(boxes[class_idx], boxes[category_idx])
                   if len(category_idx) else torch.zeros((len(class_idx), 0)))
        for row, i in enumerate(class_idx.tolist()):
            if overlap.shape[1] and overlap[row].max() > SAME_OBJECT_IOU:
                j = int(overlap[row].argmax())
                keep_scores[j] = torch.maximum(keep_scores[j], scores[i])
            else:
                keep_boxes.append(boxes[i])
                keep_scores.append(scores[i])
    if not keep_boxes:
        return torch.zeros((0, 4)), torch.zeros(0)
    boxes, scores = torch.stack(keep_boxes), torch.stack(keep_scores)
    order = torch.argsort(scores, descending=True)
    return boxes[order], scores[order]


def iou(a, b):
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = (max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) +
             max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - intersection)
    return 0.0 if union <= 0.0 else intersection / union


def best_match(track_box, rows, assigned, threshold):
    """The unassigned detection that overlaps the track the most (IoU >= threshold)."""
    best, best_index = 0.0, None
    for i, row in enumerate(rows):
        if i in assigned:
            continue
        overlap = iou(track_box, row[:4])
        if overlap > best:
            best, best_index = overlap, i
    return None if best < threshold else best_index
