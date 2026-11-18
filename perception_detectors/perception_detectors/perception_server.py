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

"""Perception server: runs the detection models of the perception system (its remote engine) on
another computer, typically one with a GPU. No ROS needed.

    python3 -m perception_detectors.perception_server --port 8765 --preload omdet
    # in a ROS workspace: ros2 run perception_detectors perception_server --port 8765

By default it listens only on 127.0.0.1: reach it through an SSH tunnel from the robot's
computer (ssh -N -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes
-L 8765:localhost:8765 <user>@<server>). --host 0.0.0.0 opens it to the network; then use
--token. How to install it on a server: docs/PERCEPTION_SERVER.md. The protocol is in
protocol.py.
"""

import argparse
import json
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from perception_detectors import backends, protocol
from perception_detectors.pipeline import DetectionPipeline
from perception_detectors.tracking import resolve_device, StrongSortTracker

SESSION_IDLE_S = 600.0   # sessions without requests for this long are closed


def log(message):
    """Print, but never fail: the terminal that started the server may be gone."""
    try:
        print(message, flush=True)
    except (BrokenPipeError, OSError):
        pass


class Server:

    def __init__(self, device, token=None):
        self.device = resolve_device(device)
        self.token = token
        self.models = {}            # (name, options) -> backend
        self.locks = {}             # (name, options) -> lock: one inference at a time
        self.sessions = {}          # id -> {'pipeline', 'key', 'last'}
        self.lock = threading.Lock()

    def model(self, name, options):
        key = (name, json.dumps(options, sort_keys=True))
        with self.lock:
            if key not in self.locks:
                self.locks[key] = threading.Lock()
        with self.locks[key]:
            if key not in self.models:
                start = time.monotonic()
                self.models[key] = backends.create(name, self.device, options)
                log(f'Model {name} {options or ""} loaded on {self.device} '
                      f'in {time.monotonic() - start:.1f} s')
        return key, self.models[key]

    def open_session(self, body):
        start = time.monotonic()
        key, backend = self.model(body.get('model', 'omdet'), body.get('options') or {})
        tracker = None
        if body.get('tracker') is not None:
            tracker = StrongSortTracker(body['tracker'])
        session = secrets.token_hex(8)
        with self.lock:
            self.close_idle()
            self.sessions[session] = {'pipeline': DetectionPipeline(backend, tracker),
                                      'key': key, 'last': time.monotonic()}
        log(f'Session {session}: model {key[0]}, tracking {tracker is not None}')
        return {'session': session, 'load_s': round(time.monotonic() - start, 2)}

    def session(self, session_id):
        with self.lock:
            entry = self.sessions.get(session_id)
            if entry is not None:
                entry['last'] = time.monotonic()
        return entry

    def close_idle(self):
        now = time.monotonic()
        for session_id in [s for s, e in self.sessions.items() if now - e['last'] > SESSION_IDLE_S]:
            del self.sessions[session_id]

    def health(self):
        return {'status': 'ok', 'models': list(backends.BACKENDS),
                'loaded': sorted({name for name, _ in self.models}),
                'device': str(self.device), 'sessions': len(self.sessions)}


def make_handler(server):

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'   # persistent connections

        def log_message(self, fmt, *args):
            pass  # one line per image would flood the terminal

        def reply(self, status, body):
            data = json.dumps(body).encode()
            try:
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the client gave up (its timeout): nothing to do

        def body(self):
            length = int(self.headers.get('Content-Length') or 0)
            return json.loads(self.rfile.read(length)) if length else {}

        def authorised(self):
            if server.token and self.headers.get('Authorization') != f'Bearer {server.token}':
                self.reply(401, {'error': 'missing or wrong token'})
                return False
            return True

        def handle_request(self, method):
            # The body is read first, whatever the answer: a reply that leaves it unread (an
            # unknown session, a missing token) would desynchronise the persistent connection
            try:
                body = self.body()
            except ValueError:
                return self.reply(400, {'error': 'the body is not JSON'})
            if not self.authorised():
                return
            parts = [p for p in self.path.split('?')[0].split('/') if p]
            try:
                if method == 'GET' and parts == ['health']:
                    return self.reply(200, server.health())
                if method == 'POST' and parts == ['sessions']:
                    return self.reply(200, server.open_session(body))
                if method == 'POST' and parts == ['models']:
                    _, backend = server.model(body.get('model', 'omdet'), body.get('options') or {})
                    return self.reply(200, {'open_vocabulary': backend.open_vocabulary,
                                            'classes': backend.classes()})
                if len(parts) >= 2 and parts[0] == 'sessions':
                    entry = server.session(parts[1])
                    if entry is None:
                        return self.reply(404, {'error': f'unknown session {parts[1]}'})
                    if method == 'DELETE' and len(parts) == 2:
                        with server.lock:
                            server.sessions.pop(parts[1], None)
                        return self.reply(200, {})
                    if method == 'POST' and parts[2:] == ['class']:
                        if 'labels' in body:
                            entry['pipeline'].set_labels(body['labels'])
                            return self.reply(200, {'message': f'Classes: {body["labels"]}'})
                        message = entry['pipeline'].set_class(
                            body.get('class_name', ''), body.get('category', ''))
                        return self.reply(200, {'message': message})
                    if method == 'POST' and parts[2:] == ['detect']:
                        image = protocol.decode_image(body['image'])
                        start = time.monotonic()
                        with server.locks[entry['key']]:
                            detections = entry['pipeline'].process(image)
                        return self.reply(200, {
                            'detections': [d.to_dict() for d in detections],
                            'inference_ms': round((time.monotonic() - start) * 1000.0, 1)})
                self.reply(404, {'error': f'unknown request {method} {self.path}'})
            except Exception as exc:  # noqa: B902 - report any failure to the client
                self.reply(500, {'error': f'{type(exc).__name__}: {exc}'})

        def do_GET(self):  # noqa: N802
            self.handle_request('GET')

        def do_POST(self):  # noqa: N802
            self.handle_request('POST')

        def do_DELETE(self):  # noqa: N802
            self.handle_request('DELETE')

    return Handler


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--host', default='127.0.0.1',
                        help='127.0.0.1 (default: only through an SSH tunnel) or 0.0.0.0')
    parser.add_argument('--port', type=int, default=protocol.DEFAULT_PORT)
    parser.add_argument('--device', default='', help='cuda, cuda:1, cpu (default: cuda if any)')
    parser.add_argument('--token', default='',
                        help='Require "Authorization: Bearer <token>" (use it with --host 0.0.0.0)')
    parser.add_argument('--preload', nargs='*', default=[],
                        help='Models to load at start, e.g. omdet grounding_dino')
    parsed, _ = parser.parse_known_args(args)  # ros2 run adds --ros-args
    server = Server(parsed.device, parsed.token or None)
    httpd = ThreadingHTTPServer((parsed.host, parsed.port), make_handler(server))
    # Listen first, load afterwards: clients can connect (and wait) meanwhile
    threading.Thread(target=lambda: [server.model(name, {}) for name in parsed.preload],
                     daemon=True).start()
    log(f'Perception server on http://{parsed.host}:{parsed.port} (device {server.device}, '
          f'models {", ".join(backends.BACKENDS)})')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == '__main__':
    main()
