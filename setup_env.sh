#!/bin/bash
# YOLO + BLIP2 + LiDAR 위험도 평가 노드 환경 설정

# ROS2 환경 설정
source /opt/ros/humble/setup.bash
source /root/ws/install/setup.bash

# Python 경로 설정
export PYTHONPATH="/root/ws/src/risk_nav/src:$PYTHONPATH"

# GPU 설정 (CUDA 사용시)
export CUDA_VISIBLE_DEVICES=0

echo "YOLO + BLIP2 + LiDAR 위험도 평가 노드 환경 설정 완료"
