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

"""SAM 3 (Meta, "SAM License", gated weights on Hugging Face): detects and segments every
instance of a noun phrase. Much stronger than OMDet, but it needs a server GPU, Python
3.12, PyTorch >= 2.7 and the `sam3` package (github.com/facebookresearch/sam3), and access
to facebook/sam3 approved on Hugging Face (`hf auth login`). See
docs/PERCEPTION_SERVER.md.

It is used image by image (its video predictor works on files, not on a stream); the
identities come from the tracker of the pipeline, as with the other models. Each label is
a separate text prompt.

Options:
  threshold   minimum score (default 0.4)

Written from the documented API (Sam3Processor.set_image / set_text_prompt returning
masks, boxes and scores); check it on the server, where the weights are available.
"""

import time

import numpy as np
from PIL import Image
import torch

from perception_detectors.backends.base import DetectorBackend


class Sam3Backend(DetectorBackend):
    name = 'sam3'

    def __init__(self, device, options):
        super().__init__(device, options)
        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as exc:
            raise RuntimeError(
                'SAM 3 is not installed: see "Perception server" in '
                'docs/PERCEPTION_SERVER.md') from exc
        self.threshold = float(options.get('threshold', 0.4))
        start = time.monotonic()
        self.processor = Sam3Processor(build_sam3_image_model())
        self.load_time_s = time.monotonic() - start

    @staticmethod
    def _tensor(value, shape):
        if value is None:
            return torch.zeros(shape)
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        return value.detach().float().cpu().reshape(shape)

    def detect(self, image, labels):
        state = self.processor.set_image(Image.fromarray(image))
        all_boxes, all_scores, all_labels = [], [], []
        for label in labels:
            output = self.processor.set_text_prompt(state=state, prompt=label)
            boxes = self._tensor(output.get('boxes'), (-1, 4))
            scores = self._tensor(output.get('scores'), (-1,))
            keep = scores >= self.threshold
            all_boxes.append(boxes[keep])
            all_scores.append(scores[keep])
            all_labels += [label] * int(keep.sum())
        return torch.cat(all_boxes), torch.cat(all_scores), all_labels
