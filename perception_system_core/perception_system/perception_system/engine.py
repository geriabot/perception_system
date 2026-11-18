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

"""What runs the detection models for the detection requests: on this computer (LocalEngine) or
in a perception server (RemoteEngine). No ROS here.

Each request has its own pipeline (class, category and tracker), as each session of the
perception server has. The models are shared: a model is loaded once for all the requests that
use it, and, on this computer, runs once per image for all the requests that ask it for the same
labels. A request alone gets exactly what the detector of the attention system gave."""

import json
import os
import threading
import time

from perception_detectors import backends, protocol
from perception_detectors.pipeline import Detection, DetectionPipeline
from perception_detectors.tracking import resolve_device, StrongSortTracker, TRACKER_DEFAULTS

TRACKERS = ('strongsort', 'bytetrack', 'none')


def options_key(model, options):
    return model, json.dumps(options or {}, sort_keys=True)


class RequestConfig:
    """What a detection request asks the engine for."""

    def __init__(self, model, options, tracker, class_name='', category='', labels=()):
        self.model = model
        self.options = dict(options or {})
        self.tracker = tracker                  # strongsort, bytetrack or none
        self.class_name = class_name.strip()
        self.category = category.strip()
        self.labels = [label for label in labels if label]

    def same_engine_setup(self, other):
        """Same model, options and tracker: an update keeps the tracker."""
        return other is not None and (self.model, self.options, self.tracker) == \
            (other.model, other.options, other.tracker)

    def has_class(self):
        return bool(self.labels or self.class_name)


def set_pipeline_class(pipeline, config):
    if config.labels:
        pipeline.set_labels(config.labels)
        return f'Classes: {config.labels}'
    return pipeline.set_class(config.class_name, config.category)


class LocalEngine:
    """The models on this computer."""

    def __init__(self, device='', tracker_parameters=None, log=print):
        self.device = resolve_device(device)
        self.tracker_parameters = dict(TRACKER_DEFAULTS, **(tracker_parameters or {}))
        self.log = log
        self.backends = {}        # (model, options) -> backend
        self.pipelines = {}       # request -> (key, pipeline, config)
        self.lock = threading.RLock()

    def load(self, model, options=None):
        """The backend of a model, loaded the first time."""
        key = options_key(model, options)
        with self.lock:
            if key not in self.backends:
                start = time.monotonic()
                self.backends[key] = backends.create(model, self.device, options or {})
                self.log(f'Model {model} {options or ""} loaded on {self.device} in '
                         f'{time.monotonic() - start:.1f} s')
            return key, self.backends[key]

    def make_tracker(self, tracker):
        if tracker == 'none':
            return None
        return StrongSortTracker(dict(self.tracker_parameters, tracker_type=tracker), log=self.log)

    def open(self, name, config):
        """Opens or updates the request. Returns a message for the log."""
        with self.lock:
            current = self.pipelines.get(name)
            if current is None or not config.same_engine_setup(current[2]):
                key, backend = self.load(config.model, config.options)
                pipeline = DetectionPipeline(backend, self.make_tracker(config.tracker))
            else:
                key, pipeline = current[0], current[1]
            message = set_pipeline_class(pipeline, config)
            self.pipelines[name] = (key, pipeline, config)
            return message

    def close(self, name):
        with self.lock:
            self.pipelines.pop(name, None)

    def process(self, image, names):
        """The detections of each request in an RGB image: {name: [Detection] or Exception}."""
        results = {}
        outputs = {}     # (model key, labels) -> output of the model in this image
        with self.lock:
            for name in names:
                if name not in self.pipelines:
                    continue
                key, pipeline, _ = self.pipelines[name]
                backend = self.backends[key]
                shared = (key, tuple(pipeline.labels))

                def detect(img, labels, shared=shared, backend=backend):
                    if shared not in outputs:
                        outputs[shared] = backend.detect(img, labels)
                    return outputs[shared]

                try:
                    results[name] = pipeline.process(image, detect)
                except Exception as exc:  # noqa: B902 - one request failing must not stop others
                    outputs.pop(shared, None)
                    results[name] = exc
        return results

    def models(self):
        # Without the lock: the status must not wait for an image being processed
        return sorted({model for model, _ in list(self.backends)})

    def available(self):
        return list(backends.BACKENDS)

    def model_info(self, model, options=None):
        """(open_vocabulary, classes) of a model, loading it if needed."""
        _, backend = self.load(model, options)
        return backend.open_vocabulary, backend.classes()

    def shutdown(self):
        with self.lock:
            self.pipelines.clear()
            self.backends.clear()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class RemoteEngine:
    """The models in a perception server (perception_detectors.perception_server): one session per
    request, each with its class and tracker."""

    def __init__(self, server_url, token_env='', image_format='jpeg', jpeg_quality=90,
                 timeout_s=2.0, wait_for_server_s=60.0, connect_timeout_s=600.0,
                 added_latency_ms=0.0,
                 tracker_parameters=None, log=print):
        token = os.environ.get(token_env) if token_env else None
        if token_env and not token:
            raise RuntimeError(f'The environment variable {token_env} (token_env) is empty')
        self.url = server_url
        self.client = protocol.Client(server_url, token)
        self.image_format = image_format
        self.jpeg_quality = jpeg_quality
        self.timeout_s = timeout_s
        self.wait_for_server_s = wait_for_server_s
        self.connect_timeout_s = connect_timeout_s
        self.added_latency_ms = added_latency_ms
        self.tracker_parameters = dict(TRACKER_DEFAULTS, **(tracker_parameters or {}))
        self.log = log
        self.sessions = {}      # request -> {'session', 'config'}
        self.health = {}
        self.lock = threading.RLock()

    def connect(self):
        deadline = time.monotonic() + self.wait_for_server_s
        while True:
            try:
                self.health = self.client.request('GET', '/health', timeout=10.0)
                break
            except (OSError, protocol.ServerError) as exc:
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f'Cannot reach the perception server at {self.url} ({exc}). Is it '
                        'running, and the SSH tunnel open?') from exc
                self.log(f'Waiting for the perception server at {self.url}...')
                time.sleep(1.0)
        self.log(f'Perception server at {self.url}: device {self.health.get("device")}, '
                 f'models {self.health.get("models")}')

    def load(self, model, options=None):
        """Opens and closes a session, so that the server loads the model now."""
        reply = self.client.request('POST', '/sessions', {
            'model': model, 'options': options or {}, 'tracker': None},
            timeout=self.connect_timeout_s)
        self.client.request('DELETE', f'/sessions/{reply["session"]}', timeout=5.0)
        self.log(f'Model {model} ready on the server (load {reply.get("load_s", 0):.1f} s)')

    def tracker_body(self, tracker):
        if tracker == 'none':
            return None
        return dict(self.tracker_parameters, tracker_type=tracker)

    def open_session(self, name, config):
        start = time.monotonic()
        reply = self.client.request('POST', '/sessions', {
            'model': config.model, 'options': config.options,
            'tracker': self.tracker_body(config.tracker)}, timeout=self.connect_timeout_s)
        self.sessions[name] = {'session': reply['session'], 'config': config}
        self.log(f'Request {name}: session {reply["session"]} with model {config.model} '
                 f'({time.monotonic() - start:.1f} s, model load {reply.get("load_s", 0):.1f} s)')

    def send_class(self, name):
        entry = self.sessions[name]
        config = entry['config']
        body = ({'labels': config.labels} if config.labels
                else {'class_name': config.class_name, 'category': config.category})
        return self.client.request('POST', f'/sessions/{entry["session"]}/class', body,
                                   timeout=10.0).get('message', '')

    def open(self, name, config):
        with self.lock:
            current = self.sessions.get(name)
            if current is None or not config.same_engine_setup(current['config']):
                if current is not None:
                    self.close(name)
                self.open_session(name, config)
            else:
                current['config'] = config
            try:
                return self.send_class(name)
            except (OSError, protocol.ServerError) as exc:
                return (f'Could not set the class on the server ({exc}); it will be set again '
                        'when the server answers')

    def close(self, name):
        with self.lock:
            entry = self.sessions.pop(name, None)
        if entry is None:
            return
        try:
            self.client.request('DELETE', f'/sessions/{entry["session"]}', timeout=5.0)
        except (OSError, protocol.ServerError):
            pass

    def detect(self, name, encoded):
        entry = self.sessions[name]
        try:
            reply = self.client.request('POST', f'/sessions/{entry["session"]}/detect', encoded,
                                        timeout=self.timeout_s)
        except protocol.ServerError as exc:
            if exc.status == 404:   # the server was restarted: open the session again
                self.log(f'The server lost the session of {name}: opening it again')
                self.open_session(name, entry['config'])
                self.send_class(name)
            raise
        return [Detection(**d) for d in reply['detections']]

    def process(self, image, names):
        results = {}
        with self.lock:
            names = [n for n in names if n in self.sessions]
            if not names:
                return results
            if self.added_latency_ms > 0:
                time.sleep(self.added_latency_ms / 1000.0)
            encoded = {'image': protocol.encode_image(image, self.image_format,
                                                      self.jpeg_quality),
                       'format': self.image_format}
            for name in names:
                if not self.sessions[name]['config'].has_class():
                    results[name] = []
                    continue
                try:
                    results[name] = self.detect(name, encoded)
                except Exception as exc:  # noqa: B902 - publish "nothing seen" and go on
                    results[name] = exc
        return results

    def models(self):
        return sorted({entry['config'].model for entry in list(self.sessions.values())})

    def available(self):
        return list(self.health.get('models', []))

    def model_info(self, model, options=None):
        reply = self.client.request('POST', '/models', {'model': model, 'options': options or {}},
                                    timeout=self.connect_timeout_s)
        return reply['open_vocabulary'], reply['classes']

    def shutdown(self):
        for name in list(self.sessions):
            self.close(name)
        self.client.close()
