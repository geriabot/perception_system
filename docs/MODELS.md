# Detection models and tracker

Every model goes through the same pipeline (`perception_detectors/pipeline.py`): it finds the
objects of the requested labels, the class and its category are merged, and the tracker of the
request gives the ids. The same pipeline runs in the sensing node (local engine) and in the
perception server (remote engine), so a model behaves the same on the robot and on a server.

| Model (`model`) | What | License | Where it runs | Notes |
|---|---|---|---|---|
| `omdet` (default) | OMDet-Turbo, `omlab/omdet-turbo-swin-tiny-hf` | Apache-2.0 | GPU of 4 GB; a CPU, slowly | Open vocabulary, fast; weak with attributes, so the category of the class is searched too |
| `grounding_dino` | Grounding DINO, `IDEA-Research/grounding-dino-tiny` (or `-base`) | Apache-2.0 | Server GPU | Stronger with phrases ("person in red") |
| `sam3` | SAM 3, `facebook/sam3` | SAM License, gated weights | Server GPU, its own environment | Experimental ([PERCEPTION_SERVER.md](PERCEPTION_SERVER.md#sam-3)) |
| `yolo` | YOLO11 (Ultralytics), `yolo11s.pt` (`yolo11n.pt` on a CPU) | AGPL-3.0 | Any GPU, or a CPU | Closed set: the 80 classes of COCO (`list_models`); a label that is not one of them finds nothing |
| `yolo_world` | YOLO-World, `yolov8s-worldv2.pt` | AGPL-3.0 | GPU | Open vocabulary at YOLO's speed |
| `color_blobs` | Regions of a color named in the label ("red", "person in red") | Apache-2.0 | A CPU, milliseconds | No learned model: tests and simulations without a GPU |

The options of each model (`model_options`, YAML: `threshold`, `model_id`, `imgsz`, `task`...)
are documented in its file in `perception_detectors/perception_detectors/backends/`. The weights
of the Hugging Face models are downloaded to `~/.cache/huggingface` the first time; those of
Ultralytics, to the working directory.

## Class and category

A request for the class "person in red" with the category "person" searches both texts and
publishes every detection as "person in red": open-vocabulary detectors are unreliable with
attributes, so every person is a candidate, and the client chooses among them (the attention
system asks a multimodal model, looking at `detections_image`). A box of the class over a box of
the category (IoU > 0.5) is the same object, kept once. With `labels`, several classes are
searched and each detection keeps its own.

## Tracker

| `tracker` | What | When |
|---|---|---|
| `strongsort` (default) | StrongSORT with appearance re-identification (OSNet): ids through occlusions and crossings, and for objects carried by a person | Whenever there is a GPU |
| `bytetrack` | ByteTrack, motion only: far cheaper | A CPU, many requests |
| `none` | No ids | `detect_once`, counting |

A StrongSORT id is given once a track has `min_hits` detections; until then the detection has an
empty id and is not drawn in `detections_image`. ByteTrack gives ids at once. Each request
numbers its tracks from 1.

## Adding a model

Write a class in `perception_detectors/perception_detectors/backends/` following `base.py`:
`detect(image, labels) -> (boxes, scores, labels)` (tensors in pixels, `[x0, y0, x1, y1]`),
`open_vocabulary`, and `classes()` for a closed-set model. Add it to `BACKENDS` in
`backends/__init__.py`. It then works in both engines, and its tests can use `color_blobs.py` as an
example of a minimal backend.
