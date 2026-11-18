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

"""Colored blobs: a detector without a learned model, for tests and simulations without a GPU.

A label is searched as the first color word it contains ("red", "red box", "person in red":
red); a label without a color word finds nothing. Each connected region of that color, larger
than min_area, is a detection with score 1.0. It is deterministic and runs in a few milliseconds
on a CPU, so the rest of the system (requests, tracker, 3D placement, target frames) can be
tested with exact expected outputs.

Options:
  min_area     minimum area of a region, in pixels (default 100)
  saturation   minimum HSV saturation of a colored pixel, 0-255 (default 120)
  value        minimum HSV value of a colored pixel, 0-255 (default 70)
"""

import cv2
import numpy as np
import torch

from perception_detectors.backends.base import DetectorBackend

# Hue ranges (OpenCV: 0-179) of each color word
HUES = {
    'red': [(0, 8), (172, 179)],
    'orange': [(9, 20)],
    'yellow': [(21, 34)],
    'green': [(35, 85)],
    'cyan': [(86, 100)],
    'blue': [(101, 130)],
    'purple': [(131, 155)],
    'magenta': [(156, 171)],
}


def color_of(label):
    """The first color word of a label, or None."""
    for word in label.lower().replace('-', ' ').split():
        if word in HUES:
            return word
    return None


class ColorBlobsBackend(DetectorBackend):
    name = 'color_blobs'
    open_vocabulary = False

    def classes(self):
        """Any label with one of these color words."""
        return list(HUES)

    def __init__(self, device, options):
        super().__init__(device, options)
        self.min_area = int(options.get('min_area', 100))
        self.saturation = int(options.get('saturation', 120))
        self.value = int(options.get('value', 70))
        self.load_time_s = 0.0

    def mask(self, hsv, color):
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for low, high in HUES[color]:
            mask |= cv2.inRange(hsv, (low, self.saturation, self.value), (high, 255, 255))
        return mask

    def detect(self, image, labels):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        boxes, scores, found = [], [], []
        for label in labels:
            color = color_of(label)
            if color is None:
                continue
            count, _, stats, _ = cv2.connectedComponentsWithStats(self.mask(hsv, color), 8)
            for i in range(1, count):   # 0 is the background
                x, y, w, h, area = stats[i]
                if area < self.min_area:
                    continue
                boxes.append([float(x), float(y), float(x + w), float(y + h)])
                scores.append(1.0)
                found.append(label)
        if not boxes:
            return torch.zeros((0, 4)), torch.zeros(0), []
        return torch.tensor(boxes), torch.tensor(scores), found
