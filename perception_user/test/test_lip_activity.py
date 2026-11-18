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

"""The classification of the lip activity, on synthetic lip-gap ratios at 30 images/s."""

import math

from perception_user.lip_activity import LipActivity

FPS = 30.0


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def run(ratios):
    clock = Clock()
    activity = LipActivity(clock=clock)
    states = []
    for ratio in ratios:
        speaking, _, _ = activity.update(ratio)
        states.append(speaking)
        clock.t += 1.0 / FPS
    return states


def speech(seconds, base=0.05, amplitude=0.04, hz=5.0):
    return [base + amplitude * (0.5 + 0.5 * math.sin(2 * math.pi * hz * i / FPS))
            for i in range(int(seconds * FPS))]


def still(seconds, ratio=0.05):
    return [ratio] * int(seconds * FPS)


def first_true(states):
    return next(i for i, s in enumerate(states) if s) / FPS


def test_speech_is_reported_after_the_onset():
    states = run(still(1.0) + speech(3.0))
    assert not any(states[:int(FPS)])
    assert 1.0 + 0.9 <= first_true(states) <= 1.0 + 1.3
    assert states[-1]


def test_a_single_opening_is_not_speech():
    opening = [0.05 + 0.15 * min(i / 6.0, 1.0) for i in range(int(2 * FPS))]  # opens, stays open
    assert not any(run(still(1.0) + opening + still(1.0)))


def test_silence_after_speech():
    states = run(speech(3.0) + still(3.0))
    end_of_speech = int(3.0 * FPS)
    assert states[end_of_speech]                  # still speaking just after
    last_true = max(i for i, s in enumerate(states) if s) / FPS
    assert 3.0 + 1.2 <= last_true + 1.0 / FPS <= 3.0 + 2.0
    assert not states[-1]
