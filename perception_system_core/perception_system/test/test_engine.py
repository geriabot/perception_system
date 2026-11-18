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

"""The engines: requests isolated from each other, one model run per image for requests with the
same labels, and the remote engine equal to the local one (perception server in a thread)."""

from http.server import ThreadingHTTPServer
import threading

import numpy as np
import pytest

from perception_detectors import backends, perception_server
from perception_detectors.backends.color_blobs import ColorBlobsBackend
from perception_system.engine import LocalEngine, RemoteEngine, RequestConfig

RED, BLUE = (220, 20, 20), (20, 20, 220)


def scene(objects, size=(240, 320)):
    image = np.full((*size, 3), 128, dtype=np.uint8)
    for x0, y0, x1, y1, color in objects:
        image[y0:y1, x0:x1] = color
    return image


def frames(n=10):
    for i in range(n):
        x = 20 + 5 * i
        yield scene([(x, 40, x + 40, 140, RED), (250 - x, 60, 290 - x, 160, RED),
                     (150, 10, 190, 50, BLUE)])


class CountingBlobs(ColorBlobsBackend):
    calls = 0

    def detect(self, image, labels):
        CountingBlobs.calls += 1
        return super().detect(image, labels)


@pytest.fixture
def counting(monkeypatch):
    monkeypatch.setitem(backends.BACKENDS, 'counting', 'test_engine:CountingBlobs')
    CountingBlobs.calls = 0
    return CountingBlobs


def config(class_name='', category='', labels=(), tracker='bytetrack', model='color_blobs'):
    return RequestConfig(model, {}, tracker, class_name, category, labels)


def as_tuples(detections):
    return [(d.cx, d.cy, d.w, d.h, d.score, d.label, d.id) for d in detections]


def test_requests_are_isolated():
    engine = LocalEngine('cpu', {'device': 'cpu'}, log=lambda *_: None)
    engine.open('a', config('red'))
    engine.open('b', config('blue'))
    engine.open('c', config(''))
    for image in frames(4):
        results = engine.process(image, ['a', 'b', 'c'])
    assert {d.label for d in results['a']} == {'red'} and len(results['a']) == 2
    assert {d.label for d in results['b']} == {'blue'} and len(results['b']) == 1
    assert results['c'] == []
    # Each request has its own tracker: its ids start from its own first track
    assert sorted(d.id for d in results['a']) == ['1', '2']
    assert [d.id for d in results['b']] == ['1']


def test_same_labels_run_the_model_once(counting):
    engine = LocalEngine('cpu', {'device': 'cpu'}, log=lambda *_: None)
    engine.open('a', config('red', model='counting'))
    engine.open('b', config('red', model='counting', tracker='none'))
    engine.open('c', config('blue', model='counting'))
    engine.process(next(frames(1)), ['a', 'b', 'c'])
    assert counting.calls == 2      # red once for a and b, blue for c
    assert engine.models() == ['counting']


def test_alone_or_with_others_gives_the_same():
    alone = LocalEngine('cpu', {'device': 'cpu'}, log=lambda *_: None)
    alone.open('a', config('red box', 'red'))
    shared = LocalEngine('cpu', {'device': 'cpu'}, log=lambda *_: None)
    shared.open('a', config('red box', 'red'))
    shared.open('b', config('red box', 'red', tracker='strongsort'))
    shared.open('c', config('blue'))
    for image in frames():
        assert as_tuples(alone.process(image, ['a'])['a']) == \
            as_tuples(shared.process(image, ['a', 'b', 'c'])['a'])


def test_update_keeps_the_tracker_unless_the_setup_changes():
    engine = LocalEngine('cpu', {'device': 'cpu'}, log=lambda *_: None)
    engine.open('a', config('red'))
    tracker = engine.pipelines['a'][1].tracker
    engine.open('a', config('red box', 'red'))
    assert engine.pipelines['a'][1].tracker is tracker
    engine.open('a', config('red', tracker='strongsort'))
    assert engine.pipelines['a'][1].tracker is not tracker


def test_a_failing_request_does_not_stop_the_others(monkeypatch):
    engine = LocalEngine('cpu', {'device': 'cpu'}, log=lambda *_: None)
    engine.open('a', config('red'))
    engine.open('b', config('blue'))

    def broken(image, detect=None):
        raise RuntimeError('model failure')
    monkeypatch.setattr(engine.pipelines['a'][1], 'process', broken)
    results = engine.process(next(frames(1)), ['a', 'b'])
    assert isinstance(results['a'], RuntimeError)
    assert len(results['b']) == 1


@pytest.fixture
def server():
    state = perception_server.Server('cpu')
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), perception_server.make_handler(state))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{httpd.server_address[1]}', state
    httpd.shutdown()
    httpd.server_close()


def test_remote_equals_local(server):
    url, state = server
    remote = RemoteEngine(url, image_format='png', tracker_parameters={'device': 'cpu'},
                          wait_for_server_s=5.0, log=lambda *_: None)
    remote.connect()
    local = LocalEngine('cpu', {'device': 'cpu'}, log=lambda *_: None)
    for engine in (remote, local):
        engine.open('a', config('red box', 'red', tracker='strongsort'))
        engine.open('b', config(labels=['red', 'blue'], tracker='bytetrack'))
    for image in frames():
        r, lo = remote.process(image, ['a', 'b']), local.process(image, ['a', 'b'])
        for name in ('a', 'b'):
            assert as_tuples(r[name]) == as_tuples(lo[name])
    assert len(state.sessions) == 2
    remote.close('a')
    assert len(state.sessions) == 1


def test_remote_reopens_a_lost_session(server):
    url, state = server
    remote = RemoteEngine(url, image_format='png', tracker_parameters={'device': 'cpu'},
                          wait_for_server_s=5.0, log=lambda *_: None)
    remote.connect()
    remote.open('a', config('red'))
    state.sessions.clear()                     # the server restarted
    first = remote.process(next(frames(1)), ['a'])['a']
    assert isinstance(first, Exception)        # that image is lost...
    second = remote.process(next(frames(1)), ['a'])['a']
    assert len(second) == 2                    # ...the next one, with the class set again


def test_model_information():
    engine = LocalEngine('cpu', {'device': 'cpu'}, log=lambda *_: None)
    assert 'yolo' in engine.available() and 'color_blobs' in engine.available()
    open_vocabulary, classes = engine.model_info('color_blobs')
    assert not open_vocabulary and 'red' in classes
    assert engine.models() == ['color_blobs']


def test_remote_model_information(server):
    url, _ = server
    remote = RemoteEngine(url, wait_for_server_s=5.0, log=lambda *_: None)
    remote.connect()
    assert 'color_blobs' in remote.available()
    assert remote.model_info('color_blobs') == (False, ColorBlobsBackend('cpu', {}).classes())
