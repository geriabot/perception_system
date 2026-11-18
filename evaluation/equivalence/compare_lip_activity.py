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

"""The classification of the lip activity of the perception system (perception_user) against the
one of the interaction system (interaction_perception/lip_activity_detector.py), on the same
random sequences of lip-gap ratios and the same clock: the same speaking state at every image.

    python3 compare_lip_activity.py <path of interaction_perception/interaction_perception>
"""

import collections
import importlib.util
import random
import sys
import types

import numpy as np

from perception_user.lip_activity import LipActivity


def load_old(path):
    # Only its classification: the module imports ROS and MediaPipe, which are replaced here
    for name in ('rclpy', 'rclpy.node', 'rclpy.qos', 'sensor_msgs', 'sensor_msgs.msg',
                 'std_msgs', 'std_msgs.msg', 'cv_bridge', 'mediapipe', 'mediapipe.tasks',
                 'mediapipe.tasks.python', 'mediapipe.tasks.python.vision'):
        module = types.ModuleType(name)
        module.__dict__.update({k: object for k in (
            'Node', 'HistoryPolicy', 'QoSProfile', 'ReliabilityPolicy', 'Image', 'Bool',
            'Float32', 'CvBridge')})
        sys.modules.setdefault(name, module)
    sys.modules['mediapipe'].tasks = sys.modules['mediapipe.tasks']
    sys.modules['mediapipe.tasks'].python = sys.modules['mediapipe.tasks.python']
    sys.modules['mediapipe.tasks.python'].vision = sys.modules['mediapipe.tasks.python.vision']
    spec = importlib.util.spec_from_file_location('old_lip', f'{path}/lip_activity_detector.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sequence(rng, n):
    """Silence, speech, single openings and missing faces, mixed."""
    out = []
    while len(out) < n:
        kind, length = rng.choice(['still', 'speech', 'open', 'noface']), rng.randint(5, 90)
        base = rng.uniform(0.02, 0.08)
        for i in range(length):
            if kind == 'speech':
                out.append(base + rng.uniform(0.0, 0.08))
            elif kind == 'open':
                out.append(base + 0.15 * min(i / 6.0, 1.0))
            elif kind == 'noface':
                out.append(0.0)
            else:
                out.append(base + rng.gauss(0.0, 0.001))
    return out[:n]


def main():
    old = load_old(sys.argv[1])
    clock = [0.0]
    old.time = types.SimpleNamespace(monotonic=lambda: clock[0])
    rng = random.Random(0)
    images = differences = speaking = 0
    for run in range(200):
        ratios = sequence(rng, 900)
        fps = rng.choice([10.0, 15.0, 30.0])
        clock[0] = 0.0
        new = LipActivity(clock=lambda: clock[0])
        window = collections.deque(maxlen=4)      # the old node smooths before its FSM
        fsm = old._LipFSM(7e-6, 0.9, 10, 1.2)
        for ratio in ratios:
            window.append(ratio)
            state, _ = fsm.update(float(np.mean(window)))
            new_speaking, _, _ = new.update(ratio)
            images += 1
            speaking += new_speaking
            differences += (state == old._LipFSM.SPEAKING) != new_speaking
            clock[0] += 1.0 / fps
    print(f'{images} images, {speaking} speaking, {differences} different')
    sys.exit(1 if differences else 0)


if __name__ == '__main__':
    main()
