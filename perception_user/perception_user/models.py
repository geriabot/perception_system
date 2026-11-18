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

"""The MediaPipe model files, downloaded once to ~/.cache/perception_system (or
$PERCEPTION_MODELS_DIR)."""

import os
from pathlib import Path
import urllib.request


def model_file(url, log=print):
    directory = Path(os.environ.get('PERCEPTION_MODELS_DIR',
                                    Path.home() / '.cache' / 'perception_system'))
    path = directory / url.rsplit('/', 1)[-1]
    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        log(f'Downloading {url} to {path}')
        partial = path.with_suffix(path.suffix + '.part')
        urllib.request.urlretrieve(url, partial)
        partial.rename(path)
    return str(path)
