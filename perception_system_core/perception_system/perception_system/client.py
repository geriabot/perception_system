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

"""A Python client of the perception system, for nodes and scripts.

    client = PerceptionClient(node)
    reply = client.start_detection('my_behavior', 'cup', outputs=OUTPUT_DETECTIONS_2D)
    node.create_subscription(Detection2DArray, reply.detections_2d_topic, callback, 10)
    ...
    client.stop('my_behavior')

Each method sends the request and waits for the response (timeout_s). It must not be called from a
callback of a single-threaded executor that also has to deliver the response: call it from a
thread of its own, or from a callback of a multi-threaded executor in a reentrant group (the
response arrives in another thread). The futures of the *_async methods can be used instead.
"""

import threading

from perception_system_interfaces.srv import DetectOnce, GetObjects, GetStatus, GetUserState, \
    ListModels, StartDetection, StartSkill, StartTargetFrame, StopRequest

OUTPUT_DEFAULT = StartDetection.Request.OUTPUT_DEFAULT
OUTPUT_DETECTIONS_2D = StartDetection.Request.OUTPUT_DETECTIONS_2D
OUTPUT_DETECTIONS_3D = StartDetection.Request.OUTPUT_DETECTIONS_3D
OUTPUT_DETECTIONS_IMAGE = StartDetection.Request.OUTPUT_DETECTIONS_IMAGE
OUTPUT_DEBUG_IMAGE = StartDetection.Request.OUTPUT_DEBUG_IMAGE
MODE_SINGLE = StartTargetFrame.Request.MODE_SINGLE
MODE_GROUP = StartTargetFrame.Request.MODE_GROUP
MODE_EACH = StartTargetFrame.Request.MODE_EACH


class PerceptionError(RuntimeError):
    """The perception system refused a request, or did not answer."""


class PerceptionClient:

    SERVICES = {
        'start_detection': StartDetection, 'start_target_frame': StartTargetFrame,
        'start_skill': StartSkill, 'stop': StopRequest, 'detect_once': DetectOnce,
        'get_user_state': GetUserState, 'get_status': GetStatus, 'get_objects': GetObjects,
        'list_models': ListModels,
    }

    def __init__(self, node, system='/perception_system', callback_group=None):
        self.node = node
        self.clients = {name: node.create_client(srv, f'{system}/{name}',
                                                 callback_group=callback_group)
                        for name, srv in self.SERVICES.items()}

    def call_async(self, name, request):
        return self.clients[name].call_async(request)

    def call(self, name, request, timeout_s=10.0, check=True):
        client = self.clients[name]
        if not client.wait_for_service(timeout_sec=timeout_s):
            raise PerceptionError(f'{client.srv_name} is not available (is the perception '
                                  'system active?)')
        done = threading.Event()
        future = client.call_async(request)
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout_s):
            future.cancel()
            raise PerceptionError(f'No response from {client.srv_name} within {timeout_s} s')
        response = future.result()
        if check and hasattr(response, 'success') and not response.success:
            raise PerceptionError(response.message)
        return response

    def start_detection(self, name, class_name='', category='', labels=(), output_namespace='',
                        outputs=OUTPUT_DEFAULT, model='', model_options='', tracker='',
                        timeout_s=30.0):
        return self.call('start_detection', StartDetection.Request(
            name=name, class_name=class_name, category=category, labels=list(labels),
            output_namespace=output_namespace, outputs=outputs, model=model,
            model_options=model_options, tracker=tracker), timeout_s)

    def start_target_frame(self, name, detection_request, frame_id, class_name, detection_id='',
                           n_detections=0, mode=MODE_SINGLE, fixed_frame_id='',
                           half_width_topic='', timeout_s=10.0):
        return self.call('start_target_frame', StartTargetFrame.Request(
            name=name, detection_request=detection_request, frame_id=frame_id,
            fixed_frame_id=fixed_frame_id, mode=mode, class_name=class_name, id=detection_id,
            n_detections=n_detections, half_width_topic=half_width_topic), timeout_s)

    def start_skill(self, name, skill, timeout_s=30.0):
        return self.call('start_skill', StartSkill.Request(name=name, skill=skill), timeout_s)

    def stop(self, name, timeout_s=10.0, check=False):
        return self.call('stop', StopRequest.Request(name=name), timeout_s, check)

    def detect_once(self, class_name='', category='', labels=(), model='', timeout_s=5.0):
        return self.call('detect_once', DetectOnce.Request(
            class_name=class_name, category=category, labels=list(labels), model=model,
            timeout_s=timeout_s), timeout_s + 5.0)

    def objects(self, request, only_tracked=False, timeout_s=2.0):
        return self.call('get_objects', GetObjects.Request(request=request,
                                                           only_tracked=only_tracked), timeout_s)

    def models(self, model='', timeout_s=60.0):
        return self.call('list_models', ListModels.Request(model=model), timeout_s)

    def user_state(self, timeout_s=2.0):
        return self.call('get_user_state', GetUserState.Request(), timeout_s)

    def status(self, timeout_s=2.0):
        return self.call('get_status', GetStatus.Request(), timeout_s).status
