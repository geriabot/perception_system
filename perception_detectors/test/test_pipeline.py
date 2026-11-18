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

"""The pipeline of a detection request with the color_blobs backend: classes, categories, the
tracker, and the marked image."""

import numpy as np
import pytest
import torch

from perception_detectors import backends, marking
from perception_detectors.backends.color_blobs import color_of
from perception_detectors.pipeline import DetectionPipeline
from perception_detectors.tracking import StrongSortTracker, TRACKER_DEFAULTS, TRACKER_TYPES

RED, BLUE, GREEN = (220, 20, 20), (20, 20, 220), (20, 200, 20)


def scene(objects, size=(240, 320)):
    """An RGB image, gray, with filled rectangles (x0, y0, x1, y1, color)."""
    image = np.full((*size, 3), 128, dtype=np.uint8)
    for x0, y0, x1, y1, color in objects:
        image[y0:y1, x0:x1] = color
    return image


@pytest.fixture
def blobs():
    return backends.create('color_blobs', 'cpu', {})


def test_color_words():
    assert color_of('person in red') == 'red'
    assert color_of('Blue-box') == 'blue'
    assert color_of('person') is None


def test_backend_finds_each_region(blobs):
    image = scene([(10, 10, 50, 60, RED), (100, 20, 140, 90, RED), (200, 50, 240, 100, BLUE)])
    boxes, scores, labels = blobs.detect(image, ['red', 'blue thing', 'person'])
    assert labels == ['red', 'red', 'blue thing']
    assert boxes.tolist() == [[10, 10, 50, 60], [100, 20, 140, 90], [200, 50, 240, 100]]
    assert scores.tolist() == [1.0, 1.0, 1.0]


def test_small_regions_ignored(blobs):
    boxes, _, labels = blobs.detect(scene([(10, 10, 15, 15, RED)]), ['red'])
    assert labels == [] and boxes.shape == (0, 4)


def test_class_and_category_published_as_the_class(blobs):
    # "red box" and its category "red" find the same region: one detection, with the class
    pipeline = DetectionPipeline(blobs)
    assert pipeline.set_class('red box', 'red') == 'New class to detect: red box ' \
        '(also looking for red)'
    detections = pipeline.process(scene([(10, 10, 50, 60, RED), (100, 20, 140, 90, RED)]))
    assert [d.label for d in detections] == ['red box', 'red box']
    assert [(d.cx, d.cy, d.w, d.h) for d in detections] == [(30, 35, 40, 50), (120, 55, 40, 70)]


def test_empty_class_detects_nothing(blobs):
    pipeline = DetectionPipeline(blobs)
    assert pipeline.set_class('', '') == 'Detection class cleared'
    assert pipeline.process(scene([(10, 10, 50, 60, RED)])) == []


def test_labels_keep_their_own_label(blobs):
    pipeline = DetectionPipeline(blobs)
    pipeline.set_labels(['red', 'blue'])
    detections = pipeline.process(scene([(10, 10, 50, 60, RED), (200, 50, 240, 100, BLUE)]))
    assert sorted(d.label for d in detections) == ['blue', 'red']


def test_shared_model_output_is_not_modified(blobs):
    """process(image, detect) with an output shared by several requests leaves it unchanged."""
    image = scene([(10, 10, 50, 60, RED), (12, 12, 48, 58, RED)])
    output = blobs.detect(image, ['red box', 'red'])
    copy = (output[0].clone(), output[1].clone(), list(output[2]))
    pipeline = DetectionPipeline(blobs)
    pipeline.set_class('red box', 'red')
    pipeline.process(image, lambda img, labels: output)
    assert torch.equal(output[0], copy[0]) and torch.equal(output[1], copy[1])
    assert output[2] == copy[2]


@pytest.mark.parametrize('tracker_type', list(TRACKER_TYPES))
def test_tracker_ids_confirmed_and_stable(blobs, tracker_type):
    params = dict(TRACKER_DEFAULTS, tracker_type=tracker_type, device='cpu')
    pipeline = DetectionPipeline(blobs, StrongSortTracker(params))
    pipeline.set_class('red', '')
    ids = []
    for frame in range(12):
        x = 20 + 4 * frame
        detections = pipeline.process(scene([(x, 40, x + 40, 140, RED),
                                             (240 - x, 60, 280 - x, 160, RED)]))
        ids.append(sorted(d.id for d in detections))
    # StrongSORT: empty until the track is confirmed (min_hits); ByteTrack confirms at once.
    # Then the same two ids
    if tracker_type == 'strongsort':
        assert ids[0] == ['', '']
    confirmed = [i for i in ids if '' not in i]
    assert confirmed and all(i == confirmed[0] for i in confirmed), ids
    assert len(set(confirmed[0])) == 2


@pytest.mark.parametrize('tracker_type', ['bytetrack', 'botsort', 'ocsort', 'deepocsort'])
def test_each_tracker_numbers_its_own_tracks(blobs, tracker_type):
    """Two requests with the same tracker: the ids of one do not depend on the other (the
    trackers whose counter boxmot shares between instances keep their own)."""
    params = dict(TRACKER_DEFAULTS, tracker_type=tracker_type, device='cpu')
    first = DetectionPipeline(blobs, StrongSortTracker(params))
    second = DetectionPipeline(blobs, StrongSortTracker(params))
    first.set_class('red', '')
    second.set_class('red', '')
    for frame in range(8):
        x = 20 + 4 * frame
        both = scene([(x, 40, x + 40, 140, RED), (240 - x, 60, 280 - x, 160, RED)])
        ids_first = sorted(d.id for d in first.process(both))
        ids_second = sorted(d.id for d in second.process(both))
    assert ids_first == ids_second and '' not in ids_first, (ids_first, ids_second)


def test_marked_image_draws_only_tracked(blobs):
    pipeline = DetectionPipeline(blobs)
    pipeline.set_class('red', '')
    image = scene([(10, 10, 50, 60, RED)])
    detections = pipeline.process(image)
    assert np.array_equal(marking.marked_image(image, detections), image)   # no id: not drawn
    detections[0].id = '7'
    assert not np.array_equal(marking.marked_image(image, detections), image)
