# Copyright 2024
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

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    map_name = LaunchConfiguration('map_name', default='my_map')
    save_map_timeout = LaunchConfiguration('save_map_timeout', default='2.0')
    free_thresh_default = LaunchConfiguration('free_thresh_default', default='0.25')
    occupied_thresh_default = LaunchConfiguration('occupied_thresh_default', default='0.65')
    map_subscribe_transient_local = LaunchConfiguration('map_subscribe_transient_local', default='true')

    return LaunchDescription([
        DeclareLaunchArgument(
            'map_name',
            default_value=map_name,
            description='Name of the map to save'),

        DeclareLaunchArgument(
            'save_map_timeout',
            default_value=save_map_timeout,
            description='Timeout for saving map'),

        DeclareLaunchArgument(
            'free_thresh_default',
            default_value=free_thresh_default,
            description='Free threshold for map saving'),

        DeclareLaunchArgument(
            'occupied_thresh_default',
            default_value=occupied_thresh_default,
            description='Occupied threshold for map saving'),

        DeclareLaunchArgument(
            'map_subscribe_transient_local',
            default_value=map_subscribe_transient_local,
            description='Map subscribe transient local'),

        Node(
            package='nav2_map_server',
            executable='map_saver_cli',
            name='map_saver_cli',
            output='screen',
            arguments=['--map-name', map_name,
                       '--save-map-timeout', save_map_timeout,
                       '--free-thresh-default', free_thresh_default,
                       '--occupied-thresh-default', occupied_thresh_default,
                       '--map-subscribe-transient-local', map_subscribe_transient_local]),
    ]) 