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

"""The perception system: the orchestrator (/perception_orchestrator, run by mission_executor of
koch2, with orchestrator:=true), /perception_system and, with user:=true, the
nodes of the skills about the user (lip_activity_detector, emotion_detector), which
/perception_system activates when a client asks for them. The orchestrator activates
/perception_system in cascade; without it, /perception_system activates itself (autostart).

    ros2 launch perception_system perception_system.launch.py camera:=rgbd use_sim_time:=true

Parameters: config/perception_system.yaml, then config/cameras/<camera>.yaml, then the file of
params_file, then the launch arguments that are not empty. Every argument:
docs/LAUNCH_ARGUMENTS.md.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node

ARGUMENTS = [
    ('camera', 'none', 'The camera of the robot, config/cameras/<camera>.yaml: rgbd, rgbd_body, '
                       'astra, nao or none (no 3D placement)'),
    ('params_file', '', 'A YAML file with parameters of /perception_system, loaded after the '
                        'camera (e.g. a robot of your own)'),
    ('image_topic', '', 'The camera image (default /image_rgb)'),
    ('engine', '', 'local (the models on this computer) or remote (in a perception server)'),
    ('model', '', 'Default detection model: omdet, grounding_dino, sam3, yolo, yolo_world, '
                  'color_blobs'),
    ('model_options', '', 'Options of the default model, as YAML, e.g. "{threshold: 0.35}"'),
    ('device', '', 'local: torch device (empty: cuda if available)'),
    ('tracker', '', 'Default tracker of the requests: strongsort, bytetrack or none'),
    ('preload_model', '', 'Load the default model on activation: true or false'),
    ('zoom_small_detections', '', 'detections_image: add the small detections enlarged'),
    ('server_url', '', 'remote: the perception server (default http://localhost:8765)'),
    ('token_env', '', 'remote: environment variable with the token of the server'),
    ('server_timeout_s', '', 'remote: maximum time per image (default 2.0)'),
    ('start_perception_server', 'false', 'remote: also start a perception server on this '
                                         'computer, on the port of server_url (tests)'),
    ('camera_intrinsics', '', 'Fixed intrinsics "[fx, fy, cx, cy]" instead of the CameraInfo'),
    ('depth_estimator', '', 'The 3D placement: depth_image, ray, box_size, depth_anything or '
                            'none (overrides the camera)'),
    ('fixed_frame_id', '', 'The parent of the target frames (default odom)'),
    ('user', 'false', 'Start the nodes of the skills about the user (perception_user)'),
    ('autostart_skills', '', 'Skills active from the start, "speaking,emotion"'),
    ('startup_requests', '', 'Detection requests opened on activation, without a class, '
                             '"name" or "name:output_namespace", comma-separated ("attention:/")'),
    ('orchestrator', 'true', 'Start the orchestrator (the perception tasks: TrackObject, '
                             'WaitForObject, ObserveClass), which activates /perception_system'),
    ('slots_per_behavior', '3', 'Orchestrator: goals of each task at the same time'),
    ('autostart', '', 'Configure and activate /perception_system by itself at start (default: '
                      'true without the orchestrator, false with it)'),
    ('use_sim_time', 'false', 'Use the simulation clock'),
    ('log_level', 'info', 'Log level of the nodes'),
]

# Launch argument -> parameter, with its type
OVERRIDES = {
    'image_topic': ('image_topic', str), 'engine': ('engine', str), 'model': ('model', str),
    'model_options': ('model_options', str), 'device': ('device', str),
    'tracker': ('tracker', str), 'preload_model': ('preload_model', bool),
    'zoom_small_detections': ('zoom_small_detections', bool),
    'server_url': ('remote.server_url', str), 'token_env': ('remote.token_env', str),
    'server_timeout_s': ('remote.timeout_s', float),
    'depth_estimator': ('projection.estimator', str),
    'fixed_frame_id': ('fixed_frame_id', str),
}


def as_bool(text):
    return text.strip().lower() in ('true', '1', 'yes')


def launch_setup(context):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    share = get_package_share_directory('perception_system')
    camera_file = os.path.join(share, 'config', 'cameras', f'{arg("camera")}.yaml')
    if not os.path.exists(camera_file):
        cameras = sorted(f[:-5] for f in os.listdir(os.path.join(share, 'config', 'cameras')))
        raise ValueError(f'Unknown camera "{arg("camera")}": one of {", ".join(cameras)}')
    parameters = [os.path.join(share, 'config', 'perception_system.yaml'), camera_file]
    if arg('params_file'):
        parameters.append(arg('params_file'))
    orchestrator = as_bool(arg('orchestrator'))
    autostart = as_bool(arg('autostart')) if arg('autostart') else not orchestrator
    overrides = {'use_sim_time': as_bool(arg('use_sim_time')), 'autostart': autostart}
    for name, (parameter, kind) in OVERRIDES.items():
        value = arg(name)
        if value:
            overrides[parameter] = as_bool(value) if kind is bool else kind(value)
    if arg('camera_intrinsics'):
        import yaml
        overrides['projection.camera_intrinsics'] = [
            float(v) for v in yaml.safe_load(arg('camera_intrinsics'))]
    if arg('startup_requests'):
        overrides['startup_requests'] = [s.strip() for s in arg('startup_requests').split(',')
                                         if s.strip()]
    if arg('autostart_skills'):
        overrides['autostart_skills'] = [s.strip() for s in arg('autostart_skills').split(',')
                                         if s.strip()]
    parameters.append(overrides)
    log = ['--ros-args', '--log-level', arg('log_level')]

    nodes = [LifecycleNode(package='perception_system', executable='perception_system',
                           name='perception_system', namespace='', output='screen',
                           parameters=parameters, arguments=log)]
    if as_bool(arg('user')):
        image_topic = overrides.get('image_topic', '/image_rgb')
        common = {'use_sim_time': overrides['use_sim_time'], 'image_topic': image_topic}
        nodes += [
            LifecycleNode(package='perception_user', executable='lip_activity_detector',
                          name='lip_activity_detector', namespace='', output='screen',
                          parameters=[common], arguments=log),
            LifecycleNode(package='perception_user', executable='emotion_detector',
                          name='emotion_detector', namespace='', output='screen',
                          parameters=[common], arguments=log),
        ]
    if orchestrator:
        # Its runners publish on /behavior_status, which every koch2 runner of the
        # robot shares: remapped, so that other subsystems' orchestrators do not take their status
        # as their own (this orchestrator reads each runner's status directly)
        mission = os.path.join(get_package_share_directory('perception_orchestrator'), 'config',
                               'perception_mission.yaml')
        nodes.append(Node(
            package='koch2', executable='mission_executor',
            output='screen', arguments=[mission, '--ros-args', '--log-level', arg('log_level')],
            parameters=[{'use_sim_time': overrides['use_sim_time'],
                         'slots_per_behavior': int(arg('slots_per_behavior'))}],
            remappings=[('/behavior_status', '/perception_system/behavior_status')]))
    if as_bool(arg('start_perception_server')):
        url = overrides.get('remote.server_url', 'http://localhost:8765')
        port = url.rstrip('/').rsplit(':', 1)[-1] if url.count(':') > 1 else '8765'
        nodes.append(Node(package='perception_detectors', executable='perception_server',
                          output='screen', arguments=[
                              '--port', port, '--preload', overrides.get('model', 'omdet')]))
    return nodes


def generate_launch_description():
    return LaunchDescription(
        [DeclareLaunchArgument(name, default_value=default, description=description)
         for name, default, description in ARGUMENTS] + [OpaqueFunction(function=launch_setup)])
