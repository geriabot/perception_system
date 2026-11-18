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

"""Interface of a detection model."""


class DetectorBackend:
    """A model that finds, in an RGB image, the objects described by some text labels.

    detect(image, labels) returns (boxes, scores, labels):
      boxes   torch.Tensor (N, 4), float, [x0, y0, x1, y1] in pixels
      scores  torch.Tensor (N,), float, in [0, 1]
      labels  list of N strings, each one of the requested labels
    The pipeline adds the rest (the category, the tracker): a backend only detects.

    Parameters come from the `options` of the detector (a dict); each backend documents
    its own. `device` is a torch device string ('cuda', 'cuda:0', 'cpu').
    """

    name = ''
    # Whether it detects any text (True) or only the classes of its model (False)
    open_vocabulary = True

    def __init__(self, device, options):
        self.device = device
        self.options = options

    def classes(self):
        """The classes it can detect, for a closed-set model ([] for open vocabulary)."""
        return []

    def detect(self, image, labels):
        raise NotImplementedError
