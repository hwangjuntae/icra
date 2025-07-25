#!/usr/bin/env python3
"""
Risk Assessment System Launch File
YOLO+BLIP2 위험도 평가 노드와 Risk Mapping 노드를 함께 실행
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # RViz 설정 파일 경로
    rviz_config_file = os.path.join(
        get_package_share_directory('risk_nav'),
        'rviz',
        'config.rviz'
    )
    
    return LaunchDescription([
        # Launch Arguments
        DeclareLaunchArgument(
            'camera_topic',
            default_value='/Camera/rgb',
            description='카메라 이미지 토픽'
        ),
        DeclareLaunchArgument(
            'lidar_topic',
            default_value='/Lidar/laser_scan',
            description='LiDAR 스캔 토픽'
        ),
        DeclareLaunchArgument(
            'risk_assessment_topic',
            default_value='/risk_assessment/image',
            description='위험도 평가 결과 토픽'
        ),
        DeclareLaunchArgument(
            'risk_map_topic',
            default_value='/risk_map',
            description='Risk Map 토픽'
        ),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='RViz 실행 여부'
        ),

        
        # YOLO + BLIP2 위험도 평가 노드
        Node(
            package='risk_nav',
            executable='topic_yolo_blip2_lidar_risk',
            name='yolo_blip2_risk_node',
            output='screen',
            parameters=[{
                'camera_topic': LaunchConfiguration('camera_topic'),
                'lidar_topic': LaunchConfiguration('lidar_topic'),
                'risk_assessment_topic': LaunchConfiguration('risk_assessment_topic'),
            }],
            remappings=[
                ('/Camera/rgb', LaunchConfiguration('camera_topic')),
                ('/Lidar/laser_scan', LaunchConfiguration('lidar_topic')),
                ('/risk_assessment/image', LaunchConfiguration('risk_assessment_topic')),
            ]
        ),
        
        # Risk Mapping 노드
        Node(
            package='risk_nav',
            executable='risk_mapping',
            name='risk_mapping_node',
            output='screen',
            parameters=[{
                'risk_assessment_topic': LaunchConfiguration('risk_assessment_topic'),
                'lidar_topic': LaunchConfiguration('lidar_topic'),
                'risk_map_topic': LaunchConfiguration('risk_map_topic'),
            }],
            remappings=[
                ('/risk_assessment/image', LaunchConfiguration('risk_assessment_topic')),
                ('/Lidar/laser_scan', LaunchConfiguration('lidar_topic')),
                ('/risk_map', LaunchConfiguration('risk_map_topic')),
            ]
        ),
        
        # RViz 노드 (risk map 시각화용)
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file],
            output='screen'
        ),
    ]) 