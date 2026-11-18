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

"""The people of an image, with the 17 keypoints of their body and a tracker id: YOLO pose
(Ultralytics, AGPL-3.0, an optional dependency as for the yolo detector) and ByteTrack. No ROS
here.

The weights (yolo11n-pose.pt by default, real time on a CPU) are downloaded once to
~/.cache/perception_system (or $PERCEPTION_MODELS_DIR); model_id may name other weights of
Ultralytics (yolo11s-pose.pt, ...) or a file.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import urllib.request

import numpy as np

DEFAULT_MODEL = 'yolo11n-pose.pt'
RELEASES = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/'


@dataclass
class PoseDetection:
    id: str                 # tracker id ('' while the track is not confirmed)
    box: tuple              # x0, y0, x1, y1 in pixels
    score: float
    keypoints: np.ndarray   # (17, 3): x, y in pixels, score


def models_dir():
    return Path(os.environ.get('PERCEPTION_MODELS_DIR',
                               Path.home() / '.cache' / 'perception_system'))


def weights_file(model_id='', log=print):
    """The path of the weights: a file that exists, or weights of Ultralytics' releases
    downloaded once to the models directory."""
    model_id = model_id or DEFAULT_MODEL
    if os.path.exists(model_id):
        return model_id
    path = models_dir() / os.path.basename(model_id)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        log(f'Downloading {RELEASES + path.name} to {path}')
        partial = path.with_suffix(path.suffix + '.part')
        urllib.request.urlretrieve(RELEASES + path.name, partial)
        partial.rename(path)
    return str(path)


class PoseDetector:
    """YOLO pose and a ByteTrack tracker of its own."""

    def __init__(self, model_id='', device='', imgsz=640, threshold=0.4, tracking=None,
                 log=print):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError('The body detector needs Ultralytics: pip install ultralytics') \
                from exc
        from perception_detectors.tracking import resolve_device, StrongSortTracker
        self.model = YOLO(weights_file(model_id, log))
        self.device = str(resolve_device(device))
        self.imgsz = int(imgsz)
        self.threshold = float(threshold)
        params = dict(tracking or {})
        params['tracker_type'] = 'bytetrack'
        self.tracker = StrongSortTracker(params, log) if tracking is not False else None

    def detect(self, image):
        """image: RGB uint8 (H, W, 3). Returns [PoseDetection]."""
        import cv2
        result = self.model.predict(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR), conf=self.threshold, imgsz=self.imgsz,
            device=self.device, verbose=False)[0]
        if result.keypoints is None or len(result.boxes) == 0:
            if self.tracker is not None:
                self.tracker.update(np.empty((0, 6), dtype=np.float32), image)
            return []
        boxes = result.boxes.xyxy.float().cpu().numpy()
        scores = result.boxes.conf.float().cpu().numpy()
        keypoints = result.keypoints.data.float().cpu().numpy()
        if keypoints.shape[-1] == 2:     # weights without keypoint scores
            keypoints = np.concatenate([keypoints, np.ones(keypoints.shape[:2] + (1,))], -1)
        detections = [PoseDetection('', tuple(float(v) for v in box), float(score), kp)
                      for box, score, kp in zip(boxes, scores, keypoints)]
        if self.tracker is not None:
            self.assign_ids(image, detections)
        return detections

    def assign_ids(self, image, detections):
        from perception_detectors.pipeline import best_match
        rows = np.array([list(d.box) + [d.score, 0.0] for d in detections],
                        dtype=np.float32).reshape(-1, 6)
        tracks = self.tracker.update(rows, image)
        if tracks is None:
            return
        assigned = set()
        for track in tracks:
            index = best_match(np.array(track[:4], dtype=np.float32), rows, assigned,
                               self.tracker.iou_threshold)
            if index is not None:
                detections[index].id = str(int(track[4]))
                assigned.add(index)
