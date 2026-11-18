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

"""YOLO through Ultralytics (pip install ultralytics; AGPL-3.0: an optional dependency, not
part of this repository). The fastest detectors: real time on a CPU or a small GPU.

yolo        Closed set: the 80 COCO categories (person, cup, bowl, book, chair, ...). With the
            category of the class this is enough for most requests: "person in red" is
            detected as "person" and the MLLM checks the color in the image. A label that is
            not a COCO category finds nothing.
yolo_world  Open vocabulary (YOLO-World): any text, at YOLO speed, weaker than OMDet or
            Grounding DINO with attributes. Needs the CLIP text encoder of Ultralytics.

Options:
  model_id    weights (yolo: yolo11s.pt, also yolo11n.pt for a CPU; yolo_world:
              yolov8s-worldv2.pt), downloaded by Ultralytics the first time
  threshold   minimum score (default 0.3)
  imgsz       inference size in pixels (default 640)

Set YOLO_AUTOINSTALL=false to stop Ultralytics from installing missing packages by itself.
"""

import time

import cv2
import torch

from perception_detectors.backends.base import DetectorBackend


def _load(model_id):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError('YOLO needs Ultralytics: pip install ultralytics') from exc
    return YOLO(model_id)


class YoloBackend(DetectorBackend):
    name = 'yolo'
    default_model = 'yolo11s.pt'
    open_vocabulary = False

    def __init__(self, device, options):
        super().__init__(device, options)
        self.threshold = float(options.get('threshold', 0.3))
        self.imgsz = int(options.get('imgsz', 640))
        start = time.monotonic()
        self.model = _load(options.get('model_id') or self.default_model)
        self.load_time_s = time.monotonic() - start

    def classes(self):
        return [self.model.names[i] for i in sorted(self.model.names)]

    def classes_for(self, labels):
        """Ids of the model's classes for the labels, and the label of each id."""
        names = {name.lower(): i for i, name in self.model.names.items()}
        wanted = {names[label.lower()]: label for label in labels if label.lower() in names}
        return wanted

    def detect(self, image, labels):
        wanted = self.classes_for(labels)
        if not wanted:
            return torch.zeros((0, 4)), torch.zeros(0), []
        result = self.model.predict(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR), conf=self.threshold, imgsz=self.imgsz,
            classes=list(wanted), device=str(self.device), verbose=False)[0]
        boxes = result.boxes
        ids = [int(c) for c in boxes.cls.tolist()]
        return boxes.xyxy.float().cpu(), boxes.conf.float().cpu(), [wanted[i] for i in ids]


class YoloWorldBackend(YoloBackend):
    name = 'yolo_world'
    default_model = 'yolov8s-worldv2.pt'
    open_vocabulary = True

    def classes(self):
        return []

    def __init__(self, device, options):
        super().__init__(device, options)
        self.vocabulary = None

    def classes_for(self, labels):
        # The vocabulary is the requested labels: set it when they change
        if self.vocabulary != labels:
            self.model.set_classes(list(labels))
            self.vocabulary = list(labels)
        return dict(enumerate(labels))
