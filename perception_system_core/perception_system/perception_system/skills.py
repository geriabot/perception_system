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

"""The skills of the perception system that run in nodes of their own (e.g. the lip activity of
perception_user): lifecycle nodes, activated in cascade with the first request that needs them
and deactivated after the last one ends, so that they use no computer while nobody needs them."""

import threading

from lifecycle_msgs.msg import State, Transition
from lifecycle_msgs.srv import ChangeState, GetState


class Skill:

    def __init__(self, name, node_name, topics):
        self.name = name
        self.node_name = node_name.strip('/')
        self.topics = topics
        self.change_state = None
        self.get_state = None


def call(client, request, timeout_s):
    """A service call from a callback of a multi-threaded executor: waits for the response
    without spinning (the executor delivers it in another thread). None on timeout."""
    if not client.wait_for_service(timeout_sec=min(timeout_s, 2.0)):
        return None
    done = threading.Event()
    future = client.call_async(request)
    future.add_done_callback(lambda _: done.set())
    if not done.wait(timeout_s):
        future.cancel()
        return None
    return future.result()


class SkillManager:

    def __init__(self, node, names, nodes, topics, callback_group, timeout_s=10.0):
        if not (len(names) == len(nodes) == len(topics)):
            raise ValueError('skill_names, skill_nodes and skill_topics must have one entry per '
                             'skill')
        self.node = node
        self.timeout_s = timeout_s
        self.skills = {}
        for name, node_name, topic_list in zip(names, nodes, topics):
            if not name:
                continue
            skill = Skill(name, node_name, [t.strip() for t in topic_list.split(',') if t.strip()])
            skill.change_state = node.create_client(
                ChangeState, f'/{skill.node_name}/change_state', callback_group=callback_group)
            skill.get_state = node.create_client(
                GetState, f'/{skill.node_name}/get_state', callback_group=callback_group)
            self.skills[name] = skill
        self.active = set()
        self.lock = threading.Lock()

    def state(self, skill):
        response = call(skill.get_state, GetState.Request(), self.timeout_s)
        return None if response is None else response.current_state.id

    def transition(self, skill, transition_id):
        request = ChangeState.Request()
        request.transition.id = transition_id
        response = call(skill.change_state, request, self.timeout_s)
        return response is not None and response.success

    def activate(self, name):
        """Configures and activates the node of a skill. Returns (ok, message)."""
        skill = self.skills[name]
        with self.lock:
            state = self.state(skill)
            if state is None:
                return False, f'The node /{skill.node_name} of the skill {name} does not answer'
            if state == State.PRIMARY_STATE_UNCONFIGURED and \
                    not self.transition(skill, Transition.TRANSITION_CONFIGURE):
                return False, f'/{skill.node_name} could not be configured'
            if state != State.PRIMARY_STATE_ACTIVE and \
                    not self.transition(skill, Transition.TRANSITION_ACTIVATE):
                return False, f'/{skill.node_name} could not be activated'
            self.active.add(name)
            return True, f'Skill {name} active (/{skill.node_name})'

    def deactivate(self, name):
        skill = self.skills[name]
        with self.lock:
            self.active.discard(name)
            if self.state(skill) == State.PRIMARY_STATE_ACTIVE:
                self.transition(skill, Transition.TRANSITION_DEACTIVATE)
