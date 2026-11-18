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

"""MediaPipe's import on a computer without audio (models.without_audio)."""

import builtins
import sys

from perception_user.models import without_audio


def test_a_sounddevice_that_cannot_start_is_replaced(monkeypatch):
    saved = sys.modules.pop('sounddevice', None)
    real = builtins.__import__

    def no_audio(name, *args, **kwargs):
        if name == 'sounddevice':
            raise OSError('Error initializing PortAudio: no sound server')
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', no_audio)
    try:
        without_audio()
        stub = sys.modules['sounddevice']
        assert not hasattr(stub, 'InputStream')   # empty: MediaPipe's audio is never used
        without_audio()                            # once is enough
        assert sys.modules['sounddevice'] is stub
    finally:
        sys.modules.pop('sounddevice', None)
        if saved is not None:
            sys.modules['sounddevice'] = saved
