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

"""Grounding DINO (IDEA-Research, Apache-2.0) through Hugging Face transformers: open-set
detector from free text, stronger than OMDet with phrases but slower. The tiny model
(~170M parameters) runs on a server GPU and, slowly, on a 4 GB laptop GPU or a CPU.

Options:
  model_id         Hugging Face model (default IDEA-Research/grounding-dino-tiny; also
                   IDEA-Research/grounding-dino-base)
  threshold        minimum box score (default 0.3)
  text_threshold   minimum score of the words of a label (default 0.25)
  nms_threshold    IoU of the non-maximum suppression, per label (default 0.5)
"""

import time

import torch
import torchvision
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from perception_detectors.backends.base import DetectorBackend


def closest_label(text, labels):
    """The requested label that a phrase of Grounding DINO comes from: the phrases are
    words of the prompt, possibly only some of them (e.g. "person red")."""
    text = text.strip().lower()
    for label in labels:
        if text == label.lower():
            return label
    words = set(text.split())
    return max(labels, key=lambda label: (len(words & set(label.lower().split())),
                                          -abs(len(label.split()) - len(words))))


class GroundingDinoBackend(DetectorBackend):
    name = 'grounding_dino'

    def __init__(self, device, options):
        super().__init__(device, options)
        model_id = options.get('model_id') or 'IDEA-Research/grounding-dino-tiny'
        self.threshold = float(options.get('threshold', 0.3))
        self.text_threshold = float(options.get('text_threshold', 0.25))
        self.nms_threshold = float(options.get('nms_threshold', 0.5))
        start = time.monotonic()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.load_time_s = time.monotonic() - start

    def detect(self, image, labels):
        # Grounding DINO takes one text: the labels separated by dots
        text = '. '.join(label.lower() for label in labels) + '.'
        inputs = self.processor(images=image, text=text, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        height, width = image.shape[:2]
        result = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=self.threshold,
            text_threshold=self.text_threshold, target_sizes=[(height, width)])[0]
        phrases = result.get('text_labels', result.get('labels'))
        names = [closest_label(str(phrase), labels) for phrase in phrases]
        boxes, scores = result['boxes'], result['scores']
        if len(names) == 0:
            return boxes, scores, names
        # One box per object and label (Grounding DINO does no suppression)
        ids = torch.tensor([labels.index(n) for n in names], device=boxes.device)
        keep = torchvision.ops.batched_nms(boxes, scores, ids, self.nms_threshold)
        return boxes[keep], scores[keep], [names[i] for i in keep.tolist()]
