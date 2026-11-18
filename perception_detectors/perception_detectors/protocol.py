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

"""HTTP protocol between the remote engine of the perception system (client, on the robot) and
perception_server.

JSON over HTTP/1.1 (persistent connection). Images go as base64 JPEG (or PNG, lossless).

  GET    /health                      {"status": "ok", "models": [...], "loaded": [...],
                                       "device": "cuda:0", "sessions": 1}
  POST   /sessions                    {"model": "omdet", "options": {...},
                                       "tracker": {...} or null}
                                      -> {"session": "<id>", "load_s": 3.2}
  POST   /sessions/<id>/class         {"class_name": "person in red", "category": "person"}
                                      or {"labels": ["person", "chair"]} -> {"message": ...}
  POST   /sessions/<id>/detect        {"image": "<base64>", "format": "jpeg"}
                                      -> {"detections": [{cx, cy, w, h, score, label, id}],
                                          "inference_ms": 41.3}
  DELETE /sessions/<id>
  POST   /models                      {"model": "yolo", "options": {...}} (loads it)
                                      -> {"open_vocabulary": false, "classes": ["person", ...]}

Each session has its own class and tracker; the models are shared between sessions. With
--token on the server, every request needs the header "Authorization: Bearer <token>".
"""

import base64
import http.client
import json
import urllib.parse

import cv2
import numpy as np

DEFAULT_PORT = 8765


def encode_image(image_rgb, fmt='jpeg', quality=90):
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if fmt == 'png':
        ok, data = cv2.imencode('.png', bgr)
    else:
        ok, data = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError('Cannot encode the image')
    return base64.b64encode(data.tobytes()).decode()


def decode_image(text):
    data = np.frombuffer(base64.b64decode(text), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError('Cannot decode the image')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class ServerError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(f'HTTP {status}: {message}')
        self.status = status


class Client:
    """Persistent HTTP connection to a perception server."""

    def __init__(self, url, token=None):
        parsed = urllib.parse.urlparse(url if '://' in url else f'http://{url}')
        if parsed.scheme not in ('http', 'https'):
            raise ValueError(f'Unsupported server URL {url}')
        self.https = parsed.scheme == 'https'
        self.host = parsed.hostname
        self.port = parsed.port or (443 if self.https else DEFAULT_PORT)
        self.prefix = parsed.path.rstrip('/')
        self.token = token
        self.connection = None

    def _connect(self, timeout):
        cls = http.client.HTTPSConnection if self.https else http.client.HTTPConnection
        self.connection = cls(self.host, self.port, timeout=timeout)

    def request(self, method, path, body=None, timeout=10.0):
        headers = {'Content-Type': 'application/json'}
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        payload = json.dumps(body).encode() if body is not None else None
        for attempt in (1, 2):
            reused = self.connection is not None
            if self.connection is None:
                self._connect(timeout)
            self.connection.timeout = timeout
            if self.connection.sock is not None:
                self.connection.sock.settimeout(timeout)
            try:
                self.connection.request(method, self.prefix + path, payload, headers)
                response = self.connection.getresponse()
                data = response.read()
                break
            except (http.client.RemoteDisconnected, ConnectionResetError, BrokenPipeError):
                # A kept-alive connection that the server had closed: once more, on a new
                # one. Never after a timeout: the server may still be working on the request
                self.close()
                if attempt == 2 or not reused:
                    raise
            except (http.client.HTTPException, OSError):
                self.close()
                raise
        try:
            result = json.loads(data) if data else {}
        except ValueError:
            result = {'error': data[:200].decode(errors='replace')}
        if response.status >= 400:
            raise ServerError(response.status, result.get('error', response.reason))
        return result

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
