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

"""Identities across images (boxmot): StrongSORT, with appearance re-identification (more
robust to occlusions and crossings), or ByteTrack, motion only (much cheaper: for computers
without a GPU, where the re-identification takes most of the time)."""

from pathlib import Path
import threading

import numpy as np

# Default parameters (those of the attention system, with which they were tuned)
TRACKER_DEFAULTS = {
    'tracker_type': 'strongsort',   # strongsort | bytetrack
    'max_age': 45,
    'min_hits': 3,
    'iou_threshold': 0.3,     # minimum IoU to give a track's id to a detection
    'min_conf': 0.25,
    'max_cos_dist': 0.25,
    'max_iou_dist': 0.875,
    'nn_budget': 80,
    'mc_lambda': 0.98,
    'ema_alpha': 0.9,
    'reid_weights': 'osnet_x0_25_msmt17.pt',   # downloaded by boxmot the first time
    'device': '',
}

# ByteTrack numbers its tracks with a counter shared by every instance in the process
# (boxmot's BaseTrack._count). Each tracker keeps its own, so that the ids of a detection request
# do not depend on the other requests (or the other sessions of a perception server): every
# tracker numbers its tracks as the only one in a new process would
_BYTETRACK_COUNTER = threading.Lock()


def resolve_device(device):
    import torch
    if device:
        return torch.device(device)
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class StrongSortTracker:
    """The tracker of the pipeline: StrongSORT or ByteTrack (tracker_type)."""

    def __init__(self, params, log=print):
        p = dict(TRACKER_DEFAULTS, **{k: v for k, v in params.items() if v not in (None, '')})
        self.iou_threshold = float(p['iou_threshold'])
        self.track_count = None       # ByteTrack only: its own track counter
        if p['tracker_type'] == 'bytetrack':
            from boxmot.trackers.bytetrack.bytetrack import ByteTrack
            self.tracker = ByteTrack(min_conf=0.1, track_thresh=float(p['min_conf']),
                                     match_thresh=0.8, track_buffer=int(p['max_age']))
            self.track_count = 0
            return
        if p['tracker_type'] != 'strongsort':
            raise ValueError(
                f'tracker_type must be strongsort or bytetrack, not {p["tracker_type"]}')
        from boxmot.trackers import StrongSort
        weights = Path(p['reid_weights'])
        if p['reid_weights'] != TRACKER_DEFAULTS['reid_weights'] and not weights.exists():
            log(f'reid_weights {weights} does not exist: using the default')
            weights = Path(TRACKER_DEFAULTS['reid_weights'])
        self.tracker = StrongSort(
            reid_weights=weights,
            device=resolve_device(p['device']),
            half=False,
            min_conf=float(p['min_conf']),
            max_cos_dist=float(p['max_cos_dist']),
            max_iou_dist=float(p['max_iou_dist']),
            max_age=int(p['max_age']),
            n_init=int(p['min_hits']),
            nn_budget=int(p['nn_budget']),
            mc_lambda=float(p['mc_lambda']),
            ema_alpha=float(p['ema_alpha']))

    def update(self, rows, image):
        """rows: float32 (N, 6) [x0, y0, x1, y1, score, class]. Returns the tracks."""
        if len(rows) == 0:
            rows = np.empty((0, 6), dtype=np.float32)
        if self.track_count is None:
            return self.tracker.update(rows, image)
        from boxmot.trackers.bytetrack.basetrack import BaseTrack
        with _BYTETRACK_COUNTER:
            shared, BaseTrack._count = BaseTrack._count, self.track_count
            try:
                return self.tracker.update(rows, image)
            finally:
                self.track_count, BaseTrack._count = BaseTrack._count, shared
