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

"""The detection models. Each one is imported only when used, so a model whose
dependencies are not installed does not prevent using the others.

To add a model: write a DetectorBackend in a module of this package and add it to
BACKENDS (name -> 'module:Class').
"""

import importlib

BACKENDS = {
    'omdet': 'perception_detectors.backends.omdet:OmDetBackend',
    'grounding_dino': 'perception_detectors.backends.grounding_dino:GroundingDinoBackend',
    'sam3': 'perception_detectors.backends.sam3:Sam3Backend',
    'yolo': 'perception_detectors.backends.yolo:YoloBackend',
    'yolo_world': 'perception_detectors.backends.yolo:YoloWorldBackend',
    'color_blobs': 'perception_detectors.backends.color_blobs:ColorBlobsBackend',
}


def create(name, device, options=None):
    """The backend `name`, loaded on `device`."""
    if name not in BACKENDS:
        raise ValueError(f'Unknown detection model "{name}". Available: {", ".join(BACKENDS)}')
    module, cls = BACKENDS[name].split(':')
    return getattr(importlib.import_module(module), cls)(device, options or {})
