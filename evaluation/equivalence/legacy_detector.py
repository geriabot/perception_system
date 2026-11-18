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

"""The detector of the attention system (attention_detectors local_detector, unchanged) with the
color_blobs model of the perception system added to its models, so that both systems can be
compared on the synthetic camera without a GPU. Needs both workspaces sourced."""

from attention_detectors import backends
from attention_detectors.local_detector import main

backends.BACKENDS['color_blobs'] = 'perception_detectors.backends.color_blobs:ColorBlobsBackend'

if __name__ == '__main__':
    main()
