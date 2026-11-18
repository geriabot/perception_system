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

"""The pose model on the images of Ultralytics (people standing, and the same image turned: people
lying). Skipped without Ultralytics or without the weights (tests download nothing)."""

import os

import numpy as np
import pytest

from perception_body import body as b
from perception_body.pose import DEFAULT_MODEL, models_dir

ultralytics = pytest.importorskip('ultralytics')
WEIGHTS = models_dir() / DEFAULT_MODEL
pytestmark = pytest.mark.skipif(not WEIGHTS.exists(), reason=f'No {WEIGHTS}')


def load(name, rotation=None):
    import cv2
    image = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), 'assets', name))
    if rotation is not None:
        image = cv2.rotate(image, rotation)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@pytest.fixture(scope='module')
def detector():
    from perception_body.pose import PoseDetector
    return PoseDetector(str(WEIGHTS), 'cpu', tracking=False)


def test_people_standing(detector):
    people = detector.detect(load('bus.jpg'))
    assert len(people) >= 3
    postures = [b.posture_2d(p.keypoints, b.BodyParameters(), p.box) for p in people
                if p.score > 0.8]
    assert len(postures) >= 3 and set(postures) == {b.STANDING}
    assert all(p.keypoints.shape == (17, 3) for p in people)


def test_people_lying(detector):
    import cv2
    people = detector.detect(load('bus.jpg', cv2.ROTATE_90_COUNTERCLOCKWISE))
    postures = [b.posture_2d(p.keypoints, b.BodyParameters(), p.box) for p in people]
    assert postures and set(postures) <= {b.LYING, b.UNKNOWN} and b.LYING in postures


def test_tracker_ids():
    from perception_body.pose import PoseDetector
    detector = PoseDetector(str(WEIGHTS), 'cpu')
    image = load('bus.jpg')
    for _ in range(3):
        people = detector.detect(image)
    ids = [p.id for p in people if p.id]
    assert len(ids) >= 3 and len(set(ids)) == len(ids)
    assert np.all([p.score > 0 for p in people])
