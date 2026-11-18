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

"""Identities across images (boxmot), one tracker per detection request:

strongsort   StrongSORT, motion and appearance re-identification (OSNet): ids through occlusions
             and crossings (the attention system's)
bytetrack    ByteTrack, motion only: much cheaper, for computers without a GPU
botsort      BoT-SORT, motion with camera motion compensation, as the tracker of YOLO
             (Ultralytics' default, without re-identification)
ocsort       OC-SORT, motion only, robust to non-linear motion and occlusions
deepocsort   Deep OC-SORT, OC-SORT with appearance re-identification (OSNet)
"""

from importlib import import_module
from pathlib import Path
import threading

import numpy as np

# Default parameters (those of the attention system, with which they were tuned)
TRACKER_DEFAULTS = {
    'tracker_type': 'strongsort',   # one of TRACKER_TYPES
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

TRACKER_TYPES = ('strongsort', 'bytetrack', 'botsort', 'ocsort', 'deepocsort')

# Some trackers number their tracks with a counter shared by every instance in the process (a
# class attribute of boxmot). Each tracker keeps its own, so that the ids of a detection request
# do not depend on the other requests (or the other sessions of a perception server): every
# tracker numbers its tracks as the only one in a new process would. (module, class, attribute)
SHARED_COUNTERS = {
    'bytetrack': ('boxmot.trackers.bytetrack.basetrack', 'BaseTrack', '_count'),
    'botsort': ('boxmot.trackers.botsort.basetrack', 'BaseTrack', '_count'),
    'ocsort': ('boxmot.trackers.ocsort.ocsort', 'KalmanBoxTracker', 'count'),
    'deepocsort': ('boxmot.trackers.deepocsort.deepocsort', 'KalmanBoxTracker', 'count'),
}
_COUNTERS_LOCK = threading.Lock()


def resolve_device(device):
    import torch
    if device:
        return torch.device(device)
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class StrongSortTracker:
    """The tracker of the pipeline, of any of TRACKER_TYPES (tracker_type); the name is kept
    from when StrongSORT was the only one."""

    def __init__(self, params, log=print):
        p = dict(TRACKER_DEFAULTS, **{k: v for k, v in params.items() if v not in (None, '')})
        self.iou_threshold = float(p['iou_threshold'])
        self.type = p['tracker_type']
        if self.type not in TRACKER_TYPES:
            raise ValueError(f'tracker_type must be one of {", ".join(TRACKER_TYPES)}, not '
                             f'{self.type}')
        self.counter = None           # (class, attribute) of a shared track counter
        self.track_count = None       # its value for this tracker
        if self.type in SHARED_COUNTERS:
            module, cls, attribute = SHARED_COUNTERS[self.type]
            self.counter = (getattr(import_module(module), cls), attribute)
        with _COUNTERS_LOCK:
            shared = self._swap_in(None)
            try:
                self.tracker = self._create(p, log)
            finally:
                self.track_count = self._swap_out(shared)

    def _swap_in(self, value):
        """Puts this tracker's count in the shared counter; returns the shared value."""
        if self.counter is None:
            return None
        cls, attribute = self.counter
        shared = getattr(cls, attribute)
        if value is not None:
            setattr(cls, attribute, value)
        return shared

    def _swap_out(self, shared):
        """Takes this tracker's count back and restores the shared value."""
        if self.counter is None:
            return None
        cls, attribute = self.counter
        own = getattr(cls, attribute)
        setattr(cls, attribute, shared)
        return own

    def _weights(self, p, log):
        weights = Path(p['reid_weights'])
        if p['reid_weights'] != TRACKER_DEFAULTS['reid_weights'] and not weights.exists():
            log(f'reid_weights {weights} does not exist: using the default')
            weights = Path(TRACKER_DEFAULTS['reid_weights'])
        return weights

    def _create(self, p, log):
        common = {'max_age': int(p['max_age']), 'min_hits': int(p['min_hits']),
                  'iou_threshold': float(p['iou_threshold'])}
        if self.type == 'bytetrack':
            from boxmot.trackers.bytetrack.bytetrack import ByteTrack
            return ByteTrack(min_conf=0.1, track_thresh=float(p['min_conf']), match_thresh=0.8,
                             track_buffer=int(p['max_age']))
        if self.type == 'botsort':
            # The configuration of Ultralytics' botsort.yaml: no re-identification, sparse
            # optical flow for the camera motion
            from boxmot.trackers import BotSort
            return BotSort(reid_weights=self._weights(p, log), device=resolve_device(p['device']),
                           half=False, track_high_thresh=float(p['min_conf']),
                           track_low_thresh=0.1, new_track_thresh=float(p['min_conf']),
                           track_buffer=int(p['max_age']), match_thresh=0.8, cmc_method='sof',
                           with_reid=False)
        if self.type == 'ocsort':
            from boxmot.trackers import OcSort
            return OcSort(min_conf=0.1, det_thresh=float(p['min_conf']), **common)
        if self.type == 'deepocsort':
            from boxmot.trackers import DeepOcSort
            return DeepOcSort(reid_weights=self._weights(p, log),
                              device=resolve_device(p['device']), half=False,
                              det_thresh=float(p['min_conf']), **common)
        from boxmot.trackers import StrongSort
        return StrongSort(
            reid_weights=self._weights(p, log),
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
        if self.counter is None:
            return self.tracker.update(rows, image)
        with _COUNTERS_LOCK:
            shared = self._swap_in(self.track_count)
            try:
                return self.tracker.update(rows, image)
            finally:
                self.track_count = self._swap_out(shared)
