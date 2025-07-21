#!/bin/bash

# YOLO + BLIP2 + LiDAR 융합 위험도 평가 노드 설치 스크립트
# 사용법: ./install.sh

set -e  # 에러 시 종료

echo "=========================================="
echo "YOLO + BLIP2 + LiDAR 위험도 평가 노드 설치 시작"
echo "=========================================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 함수 정의
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 1. 시스템 패키지 업데이트
print_status "시스템 패키지 업데이트 중..."
sudo apt update

# 2. ROS2 관련 패키지 설치
print_status "ROS2 관련 패키지 설치 중..."
sudo apt install -y \
    ros-humble-sensor-msgs \
    ros-humble-std-msgs \
    ros-humble-geometry-msgs \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-image-common \
    ros-humble-vision-msgs \
    ros-humble-rqt-image-view \
    python3-cv-bridge

# 3. 시스템 개발 도구 설치
print_status "개발 도구 설치 중..."
sudo apt install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    cmake \
    git \
    wget \
    curl

# 4. Python 가상환경 생성 (선택사항)
# print_status "Python 가상환경 생성 중..."
# python3 -m venv venv
# source venv/bin/activate

# 5. Python 패키지 설치
print_status "Python 패키지 설치 중..."
pip3 install --upgrade pip

# NumPy 호환성 문제 해결
print_status "NumPy 호환성 설정 중..."
pip3 install "numpy>=1.20.0,<2.0.0" --force-reinstall

# 나머지 패키지 설치
print_status "필수 패키지 설치 중..."
pip3 install -r requirements.txt

# 6. 모델 디렉토리 생성
print_status "모델 디렉토리 생성 중..."
mkdir -p models
chmod 755 models

# 7. 실행 권한 설정
print_status "실행 권한 설정 중..."
chmod +x src/topic_yolo_blip2_lidar_risk.py

# 8. ROS2 워크스페이스 빌드
print_status "ROS2 워크스페이스 빌드 중..."
cd ../../..
source /opt/ros/humble/setup.bash
colcon build --packages-select risk_nav

# 9. 환경 설정 파일 생성
print_status "환경 설정 파일 생성 중..."
cat > src/risk_nav/setup_env.sh << 'EOF'
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
EOF

chmod +x src/risk_nav/setup_env.sh

# 10. 테스트 스크립트 생성
print_status "테스트 스크립트 생성 중..."
cat > src/risk_nav/test_node.sh << 'EOF'
#!/bin/bash
# YOLO + BLIP2 + LiDAR 위험도 평가 노드 테스트

# 환경 설정
source /root/ws/src/risk_nav/setup_env.sh

# 노드 실행
echo "YOLO + BLIP2 + LiDAR 위험도 평가 노드 실행 중..."
echo "종료하려면 Ctrl+C를 누르세요"
python3 /root/ws/src/risk_nav/src/topic_yolo_blip2_lidar_risk.py
EOF

chmod +x src/risk_nav/test_node.sh

# 11. 시각화 도구 설치 확인
print_status "시각화 도구 설치 확인 중..."
if ! command -v ros2 &> /dev/null; then
    print_error "ROS2가 설치되지 않았습니다."
    exit 1
fi

if ! ros2 pkg list | grep -q rqt_image_view; then
    print_warning "rqt_image_view 패키지가 설치되지 않았습니다."
    sudo apt install -y ros-humble-rqt-image-view
fi

# 12. 설치 완료 메시지
echo ""
echo "=========================================="
print_status "설치 완료!"
echo "=========================================="
echo ""
echo "사용법:"
echo "1. 노드 실행:"
echo "   ./test_node.sh"
echo ""
echo "2. 결과 확인:"
echo "   ros2 run rqt_image_view rqt_image_view"
echo "   토픽: /risk_assessment/image"
echo ""
echo "3. 수동 실행:"
echo "   source setup_env.sh"
echo "   python3 src/topic_yolo_blip2_lidar_risk.py"
echo ""
echo "토픽 정보:"
echo "- 구독: /Camera/rgb (sensor_msgs/Image)"
echo "- 구독: /Lidar/laser_scan (sensor_msgs/LaserScan)"
echo "- 발행: /risk_assessment/image (sensor_msgs/Image)"
echo ""
print_status "설치 과정이 완료되었습니다."

# 13. 최종 점검
print_status "최종 설치 점검 중..."
if python3 -c "import ultralytics, torch, transformers, cv2, numpy, rclpy; print('모든 패키지 임포트 성공')" 2>/dev/null; then
    print_status "모든 필수 패키지가 정상적으로 설치되었습니다."
else
    print_error "일부 패키지 설치에 문제가 있습니다. requirements.txt를 확인하세요."
fi 