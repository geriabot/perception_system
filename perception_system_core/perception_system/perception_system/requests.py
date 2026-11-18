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

"""The open requests of the perception system, and the rules that do not need a ROS graph: names,
namespaces, outputs and conflicts between requests."""

from dataclasses import dataclass, field
import re

from perception_system.engine import RequestConfig, TRACKERS

NAME_PATTERN = re.compile(r'^[A-Za-z][A-Za-z0-9_]*$')

# StartDetection.outputs (perception_system_interfaces/srv/StartDetection)
OUTPUT_DETECTIONS_2D = 1
OUTPUT_DETECTIONS_3D = 2
OUTPUT_DETECTIONS_IMAGE = 4
OUTPUT_DEBUG_IMAGE = 8
OUTPUT_ALL = OUTPUT_DETECTIONS_2D | OUTPUT_DETECTIONS_3D | OUTPUT_DETECTIONS_IMAGE | \
    OUTPUT_DEBUG_IMAGE
OUTPUT_DEFAULT = OUTPUT_DETECTIONS_2D | OUTPUT_DETECTIONS_3D | OUTPUT_DETECTIONS_IMAGE

MODE_SINGLE = 0
MODE_GROUP = 1
MODE_EACH = 2
MODE_FOLLOW = 3


class RequestError(ValueError):
    """A request that cannot be served; the message is returned to the client."""


def check_name(name, what='request'):
    if not NAME_PATTERN.match(name or ''):
        raise RequestError(
            f'Invalid {what} name "{name}": a letter, then letters, digits or _')


def namespace_of(output_namespace, name, default_namespace):
    """The prefix of a request's topics: '' for the root."""
    if not output_namespace:
        return f'{default_namespace.rstrip("/")}/{name}'
    if output_namespace.strip() == '/':
        return ''
    namespace = '/' + output_namespace.strip().strip('/')
    if not all(NAME_PATTERN.match(part) for part in namespace.strip('/').split('/')):
        raise RequestError(f'Invalid output_namespace "{output_namespace}"')
    return namespace


def outputs_of(value):
    if value == 0:
        return OUTPUT_DEFAULT
    if value & ~OUTPUT_ALL:
        raise RequestError(f'Invalid outputs {value}: a sum of 1, 2, 4 and 8')
    return value


def topics_of(namespace, outputs):
    """{output: topic} of a request."""
    names = {OUTPUT_DETECTIONS_2D: 'detections_2d', OUTPUT_DETECTIONS_3D: 'detections_3d',
             OUTPUT_DETECTIONS_IMAGE: 'detections_image', OUTPUT_DEBUG_IMAGE: 'debug_image'}
    return {flag: f'{namespace}/{topic}' for flag, topic in names.items() if outputs & flag}


def engine_config(request, defaults):
    """The RequestConfig of a StartDetection request. defaults: model, model_options (dict),
    tracker."""
    import yaml
    tracker = (request.tracker or defaults['tracker']).strip()
    if tracker not in TRACKERS:
        raise RequestError(f'Invalid tracker "{tracker}": one of {", ".join(TRACKERS)}')
    options = defaults['model_options']
    if request.model_options:
        try:
            options = yaml.safe_load(request.model_options)
        except yaml.YAMLError as exc:
            raise RequestError(f'model_options is not valid YAML: {exc}') from exc
        if not isinstance(options, dict):
            raise RequestError('model_options must be a YAML mapping, e.g. "{threshold: 0.35}"')
    return RequestConfig(request.model or defaults['model'], options, tracker,
                         request.class_name, request.category, list(request.labels))


@dataclass
class DetectionRequest:
    name: str
    config: RequestConfig
    outputs: int
    namespace: str
    publishers: dict = field(default_factory=dict)    # output flag (or 'markers') -> publisher
    markers: int = 0                                  # spheres of the last message (RViz)
    frames: set = field(default_factory=set)          # names of the target frames that follow it
    internal: bool = False                            # a DetectOnce request
    last: object = None       # the last image processed: (header, detections, {index: 3D})

    def describe(self):
        config = self.config
        if config.labels:
            what = ', '.join(config.labels)
        elif config.class_name:
            what = config.class_name + (f' (also {config.category})' if config.category and
                                        config.category.lower() != config.class_name.lower()
                                        else '')
        else:
            what = 'no class (images only)'
        return f'{what}; model {config.model}; tracker {config.tracker}; ' \
            f'{self.namespace or "/"}'


@dataclass
class FrameRequest:
    name: str
    source: str               # the detection request it follows
    frame_id: str
    fixed_frame_id: str
    mode: int
    class_name: str
    detection_id: str
    n_detections: int
    half_width_topic: str
    target: object = None     # targets.SingleTarget or targets.GroupTarget
    timer: object = None
    publishers: dict = field(default_factory=dict)

    def describe(self):
        if self.mode == MODE_SINGLE:
            what = f'{self.class_name} {self.detection_id}'
        elif self.mode == MODE_FOLLOW:
            current = self.target.detection_id if self.target is not None else self.detection_id
            what = f'{self.class_name} {current}, followed by its appearance'
        elif self.mode == MODE_EACH:
            return f'{self.frame_id}_<id> in {self.fixed_frame_id}: each ' \
                f'{self.class_name or "detection"} of {self.source}'
        else:
            n = 'all' if self.n_detections <= 0 else str(self.n_detections)
            what = f'midpoint of {n} {self.class_name}'
        return f'{self.frame_id} in {self.fixed_frame_id}: {what} of {self.source}'


@dataclass
class SkillRequest:
    name: str
    skill: str

    def describe(self):
        return f'skill {self.skill}'


def check_conflicts(requests, name, kind, namespace=None, frame_id=None):
    """Raises RequestError if a new or updated request clashes with the open ones: a name used by
    a request of another kind, a namespace or a TF frame used by another request."""
    for other_name, other in requests.items():
        if other_name == name:
            if not isinstance(other, kind):
                raise RequestError(f'"{name}" is the name of an open request of another kind')
            continue
        if namespace is not None and isinstance(other, DetectionRequest) and \
                not other.internal and other.namespace == namespace:
            raise RequestError(
                f'The request "{other_name}" already publishes on {namespace or "/"}')
        if frame_id is not None and isinstance(other, FrameRequest) and \
                other.frame_id == frame_id:
            raise RequestError(f'The request "{other_name}" already publishes the frame '
                               f'{frame_id}')
