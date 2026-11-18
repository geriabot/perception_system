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

"""Images drawn from the detections: the marked image that the MLLM receives
(/detections_image) and a debug image."""

import cv2
import numpy as np

MAX_OUTPUT_HEIGHT = 900
MAX_OUTPUT_WIDTH = 1000


def marked_image(image, detections):
    """The RGB image with each tracked detection boxed and labeled with its id
    (set-of-mark prompting). Detections without an id (not confirmed by the tracker yet)
    are not drawn: they cannot be chosen."""
    marked = image.copy()
    height, width = marked.shape[:2]
    for det in detections:
        if not det.id:
            continue
        x0, y0 = max(int(det.cx - det.w / 2), 0), max(int(det.cy - det.h / 2), 0)
        x1, y1 = int(det.cx + det.w / 2), int(det.cy + det.h / 2)
        cv2.rectangle(marked, (x0, y0), (x1, y1), (255, 0, 0), 1)
        # Label outside the box: inside, it would cover what tells the target apart (e.g. the
        # head of a person: their age, a cap, a mask). Below the box (for people, the floor),
        # else above it, else inside at the bottom
        (tw, th), baseline = cv2.getTextSize(det.id, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        lh = th + baseline + 4
        if y1 + lh <= height:
            ly0 = y1
        elif y0 - lh >= 0:
            ly0 = y0 - lh
        else:
            ly0 = max(y1 - lh, 0)
        lx0 = min(max(x0, 0), max(width - tw - 4, 0))
        cv2.rectangle(marked, (lx0, ly0), (lx0 + tw + 4, ly0 + lh), (255, 0, 0), -1)
        cv2.putText(marked, det.id, (lx0 + 2, ly0 + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return marked


ZOOM_HEIGHT = 96        # height of each enlarged detection, in pixels
ZOOM_CONTEXT = 0.5      # margin around the box, as a fraction of its size
ZOOM_MAX_SIDE = 64      # only detections smaller than this (largest side, pixels)


def with_zoomed_detections(image, marked, detections):
    """The marked image with a strip below it: each small tracked detection enlarged, with
    its id. On a low-resolution camera (e.g. 320x240) a cup and a bowl are a few pixels
    wide, too small for the MLLM to tell apart; the enlarged view shows what the box holds.
    The crops come from the camera image (without the drawn boxes)."""
    height, width = image.shape[:2]
    tiles = []
    for det in detections:
        if not det.id or max(det.w, det.h) >= ZOOM_MAX_SIDE:
            continue
        mx, my = det.w * (0.5 + ZOOM_CONTEXT), det.h * (0.5 + ZOOM_CONTEXT)
        x0, x1 = max(int(det.cx - mx), 0), min(int(det.cx + mx), width)
        y0, y1 = max(int(det.cy - my), 0), min(int(det.cy + my), height)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        crop = image[y0:y1, x0:x1]
        scale = ZOOM_HEIGHT / crop.shape[0]
        tile = cv2.resize(crop, (max(int(crop.shape[1] * scale), 1), ZOOM_HEIGHT),
                          interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (255, 0, 0), 1)
        (tw, th), baseline = cv2.getTextSize(det.id, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(tile, (0, 0), (tw + 4, th + baseline + 4), (255, 0, 0), -1)
        cv2.putText(tile, det.id, (2, th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        tiles.append(tile)
    if not tiles:
        return marked
    gap = np.full((ZOOM_HEIGHT, 4, 3), 255, dtype=marked.dtype)
    strip = np.hstack([part for tile in tiles for part in (tile, gap)][:-1])
    if strip.shape[1] > width:   # several rows would be clearer, but one fits most cases
        strip = cv2.resize(strip, (width, max(int(strip.shape[0] * width / strip.shape[1]), 1)),
                           interpolation=cv2.INTER_AREA)
    canvas = np.full((strip.shape[0] + 4, width, 3), 255, dtype=marked.dtype)
    canvas[4:, :strip.shape[1]] = strip
    return np.vstack([marked, canvas])


def debug_image(image, detections):
    """BGR image with every detection: box, center, class, id and score."""
    image = image.copy()
    for det in detections:
        x0, y0 = int(det.cx - det.w / 2), int(det.cy - det.h / 2)
        x1, y1 = int(det.cx + det.w / 2), int(det.cy + det.h / 2)
        center = (int(round(det.cx)), int(round(det.cy)))
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.circle(image, center, 4, (0, 255, 0), -1)
        cv2.drawMarker(image, center, (0, 255, 0), markerType=cv2.MARKER_CROSS,
                       markerSize=12, thickness=2)
        cv2.putText(image, f'{det.label} ({det.id}) {det.score:.2f}', (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]
    if height > MAX_OUTPUT_HEIGHT or width > MAX_OUTPUT_WIDTH:
        image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return image
