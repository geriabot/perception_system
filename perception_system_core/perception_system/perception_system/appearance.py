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

"""The appearance of a detection, to recognize an object again when its tracker id is lost
(target frames in MODE_FOLLOW). No ROS here.

osnet  The re-identification network of StrongSORT (OSNet, boxmot): trained to tell people
       apart by their clothes and build; milliseconds per crop on a CPU.
color  A histogram of the hue and saturation of the central part of the box: no model, for
       colored objects and tests; it cannot tell apart two objects of the same color.

embed(image, box) gives a unit vector; the similarity of two is their dot product (cosine).
"""

import numpy as np

APPEARANCES = ('osnet', 'color')
# The default min_similarity of each: the cosine above which two crops are the same object
DEFAULT_MIN_SIMILARITY = {'osnet': 0.7, 'color': 0.85}


def unit(vector):
    vector = np.asarray(vector, dtype=np.float32).flatten()
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0.0 else vector


class ColorAppearance:
    name = 'color'

    def __init__(self, crop=0.6, bins=(18, 4)):
        self.crop = crop
        self.bins = bins

    def embed(self, image, box):
        """image: RGB uint8. box: x0, y0, x1, y1 (pixels)."""
        import cv2
        x0, y0, x1, y1 = box
        cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * self.crop, (y1 - y0) * self.crop
        height, width = image.shape[:2]
        left, right = int(max(0, cx - w / 2)), int(min(width, cx + w / 2))
        top, bottom = int(max(0, cy - h / 2)), int(min(height, cy + h / 2))
        if right - left < 2 or bottom - top < 2:
            return np.zeros(self.bins[0] * self.bins[1], dtype=np.float32)
        hsv = cv2.cvtColor(np.ascontiguousarray(image[top:bottom, left:right]),
                           cv2.COLOR_RGB2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, list(self.bins), [0, 180, 0, 256])
        return unit(np.sqrt(histogram))


class OsnetAppearance:
    name = 'osnet'

    def __init__(self, weights='osnet_x0_25_msmt17.pt', device=''):
        from pathlib import Path

        from boxmot.reid.core.auto_backend import ReidAutoBackend

        from perception_detectors.tracking import resolve_device
        self.model = ReidAutoBackend(weights=Path(weights), device=resolve_device(device),
                                     half=False).model

    def embed(self, image, box):
        features = self.model.get_features(np.array([box], dtype=np.float32), image)
        return unit(features[0])


def make_appearance(name, reid_weights='osnet_x0_25_msmt17.pt', device=''):
    if name == 'osnet':
        return OsnetAppearance(reid_weights, device)
    if name == 'color':
        return ColorAppearance()
    raise ValueError(f'Unknown appearance "{name}": one of {", ".join(APPEARANCES)}')
