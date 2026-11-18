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

"""The perception server over one persistent connection."""

from http.server import ThreadingHTTPServer
import threading

import numpy as np
import pytest

from perception_detectors import perception_server, protocol


@pytest.fixture
def client():
    state = perception_server.Server('cpu', token='secret')
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), perception_server.make_handler(state))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield protocol.Client(f'http://127.0.0.1:{httpd.server_address[1]}', 'secret')
    httpd.shutdown()
    httpd.server_close()


def image():
    rgb = np.full((120, 160, 3), 128, dtype=np.uint8)
    rgb[20:60, 30:70] = (220, 20, 20)
    return {'image': protocol.encode_image(rgb, 'png'), 'format': 'png'}


def test_session_detect_and_close(client):
    session = client.request('POST', '/sessions', {'model': 'color_blobs', 'tracker': None})
    client.request('POST', f'/sessions/{session["session"]}/class', {'class_name': 'red'})
    reply = client.request('POST', f'/sessions/{session["session"]}/detect', image())
    assert [(d['cx'], d['cy'], d['w'], d['h'], d['label']) for d in reply['detections']] == \
        [(50.0, 40.0, 40.0, 40.0, 'red')]
    client.request('DELETE', f'/sessions/{session["session"]}')
    assert client.request('GET', '/health')['sessions'] == 0


def test_unknown_session_keeps_the_connection_usable(client):
    """A 404 (e.g. a session closed after 10 min without requests) must not leave the body of the
    request in the connection: the next request on it would be read from the wrong place."""
    with pytest.raises(protocol.ServerError) as error:
        client.request('POST', '/sessions/unknown/detect', image())
    assert error.value.status == 404
    reply = client.request('POST', '/sessions', {'model': 'color_blobs', 'tracker': None})
    assert reply['session']


def test_token_required(client):
    client.token = 'wrong'
    with pytest.raises(protocol.ServerError) as error:
        client.request('POST', '/sessions', {'model': 'color_blobs'})
    assert error.value.status == 401
    client.token = 'secret'
    assert client.request('GET', '/health')['status'] == 'ok'
