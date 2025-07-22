#!/bin/bash
# YOLO + BLIP2 + LiDAR 위험도 평가 노드 테스트

# 환경 설정
source /root/ws/src/risk_nav/setup_env.sh

# 노드 실행
echo "YOLO + BLIP2 + LiDAR 위험도 평가 노드 실행 중..."
echo "종료하려면 Ctrl+C를 누르세요"
python3 /root/ws/src/risk_nav/src/topic_yolo_blip2_lidar_risk.py
