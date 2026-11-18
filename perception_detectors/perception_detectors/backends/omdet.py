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

"""OMDet Turbo (omlab/omdet-turbo-swin-tiny-hf, Apache-2.0): fast open-vocabulary detector,
about 150 ms per image on a 4 GB laptop GPU. The detector of the system so far.

Options:
  model_id        Hugging Face model (default omlab/omdet-turbo-swin-tiny-hf)
  threshold       minimum score (default 0.3)
  nms_threshold   IoU of the non-maximum suppression, per label (default 0.3)
  task            optional task prompt of OMDet (default: none)
"""

import time

import torch
from transformers import AutoProcessor, OmDetTurboForObjectDetection

from perception_detectors.backends.base import DetectorBackend


class OmDetBackend(DetectorBackend):
    name = 'omdet'

    def __init__(self, device, options):
        super().__init__(device, options)
        model_id = options.get('model_id') or 'omlab/omdet-turbo-swin-tiny-hf'
        self.threshold = float(options.get('threshold', 0.3))
        self.nms_threshold = float(options.get('nms_threshold', 0.3))
        self.task = options.get('task') or None
        start = time.monotonic()
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        self.model = OmDetTurboForObjectDetection.from_pretrained(model_id).to(device)
        self.load_time_s = time.monotonic() - start

    def detect(self, image, labels):
        if self.task is not None:
            inputs = self.processor(image, text=labels, task=self.task, return_tensors='pt')
        else:
            inputs = self.processor(image, text=labels, return_tensors='pt')
        for key, value in inputs.items():
            inputs[key] = value.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        height, width = image.shape[:2]
        result = self.processor.post_process_grounded_object_detection(
            outputs, target_sizes=[(height, width)], text_labels=labels,
            threshold=self.threshold, nms_threshold=self.nms_threshold)[0]
        return result['boxes'], result['scores'], list(result['text_labels'])
