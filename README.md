# YOLO + BLIP2 위험도 평가 노드

YOLOv11을 사용한 객체 탐지와 BLIP2를 사용한 위험도 평가를 수행하는 ROS2 노드입니다.

## 📋 목차
- [기능](#기능)
- [시스템 요구사항](#시스템-요구사항)
- [설치](#설치)
- [사용법](#사용법)
- [토픽 정보](#토픽-정보)
- [출력 데이터](#출력-데이터)
- [문제 해결](#문제-해결)

## 🚀 기능
- **YOLOv11** 실시간 객체 탐지
- **BLIP2** 객체 및 장면 설명 생성
- **위험도 점수** 계산 및 레벨 분류
- **바운딩 박스** 시각화
- **ROS2 토픽** 실시간 결과 전송
- **GPU 가속** 지원 (CUDA)

## 💻 시스템 요구사항
- **OS**: Ubuntu 20.04/22.04
- **ROS2**: Humble Hawksbill
- **Python**: 3.8+
- **GPU**: NVIDIA GPU (선택사항, CUDA 11.8+)
- **메모리**: 최소 8GB RAM
- **저장공간**: 최소 5GB 여유 공간

## 📦 설치

### 자동 설치 (권장)
```bash
cd /root/ws/src/risk_nav
chmod +x install.sh
./install.sh
```

### 수동 설치
1. **ROS2 패키지 설치**
   ```bash
   sudo apt update
   sudo apt install -y ros-humble-sensor-msgs ros-humble-std-msgs \
       ros-humble-geometry-msgs ros-humble-cv-bridge \
       ros-humble-rqt-image-view python3-cv-bridge
   ```

2. **Python 패키지 설치**
   ```bash
   pip3 install -r requirements.txt
   ```

3. **워크스페이스 빌드**
   ```bash
   cd /root/ws
   source /opt/ros/humble/setup.bash
   colcon build --packages-select risk_nav
   ```

## 🔧 사용법

### 1. 환경 설정
```bash
source /root/ws/src/risk_nav/setup_env.sh
```

### 2. 노드 실행 (3가지 방법)

#### 방법 1: 테스트 스크립트 사용 (권장)
```bash
cd /root/ws/src/risk_nav
./test_node.sh
```

#### 방법 2: 직접 실행
```bash
cd /root/ws
python3 src/risk_nav/src/topic_yolo_blip2_risk.py
```

#### 방법 3: ROS2 launch 사용
```bash
ros2 run risk_nav topic_yolo_blip2_risk.py
```

### 3. 결과 확인

#### 시각화된 이미지 확인
```bash
# 새 터미널에서
ros2 run rqt_image_view rqt_image_view
# 토픽 선택: /risk_assessment/image
```

#### 터미널에서 토픽 정보 확인
```bash
# 토픽 정보 확인
ros2 topic info /risk_assessment/image

# 주파수 확인
ros2 topic hz /risk_assessment/image

# 토픽 목록 확인
ros2 topic list | grep risk
```

## 📡 토픽 정보
- **구독 토픽**: `/Camera/rgb` (sensor_msgs/Image)
- **발행 토픽**: `/risk_assessment/image` (sensor_msgs/Image)

## 🎯 모델 정보
- **YOLO 모델**: YOLOv11 nano (yolo11n.pt)
- **BLIP2 모델**: Salesforce/blip-image-captioning-base
- **모델 저장 위치**: `/root/ws/src/risk_nav/models/`
- **자동 다운로드**: 첫 실행 시 자동으로 모델 다운로드

## 📊 출력 데이터

### 시각화된 이미지에 표시되는 정보:
- **🔲 바운딩 박스**: 위험도 레벨에 따른 색상 구분
  - 🟢 **녹색**: 낮은 위험 (0-30점)
  - 🟡 **노란색**: 보통 위험 (31-70점)
  - 🔴 **빨간색**: 높은 위험 (71-100점)
- **📝 객체 정보**: 클래스명, 신뢰도, 위험도 점수
- **📈 전체 위험도**: 상단에 전체 객체 수와 평균 위험도 표시
- **🎬 장면 설명**: 하단에 BLIP2가 생성한 장면 설명 표시

### 위험도 레벨
| 레벨 | 점수 범위 | 설명 | 색상 |
|------|-----------|------|------|
| **Low** | 0-30점 | 낮은 위험 | 🟢 녹색 |
| **Medium** | 31-70점 | 보통 위험 | 🟡 노란색 |
| **High** | 71-100점 | 높은 위험 | 🔴 빨간색 |

### 지원 객체
- **사무용품**: chair, desk, laptop, monitor, keyboard, mouse
- **생활용품**: bottle, cup, book, cell phone, plant
- **위험물**: scissors (위험도 높음), knife (위험도 매우 높음)
- **안전장비**: fire extinguisher (위험도 낮음)
- **기타**: COCO 데이터셋의 80개 객체 클래스

## 🛠️ 문제 해결

### 자주 발생하는 문제들

#### 1. NumPy 호환성 문제
```bash
# 해결 방법
pip3 install "numpy>=1.20.0,<2.0.0" --force-reinstall
```

#### 2. cv_bridge 오류
```bash
# ROS2 cv_bridge 재설치
sudo apt install --reinstall ros-humble-cv-bridge python3-cv-bridge
```

#### 3. CUDA 관련 오류
```bash
# CUDA 버전 확인
nvidia-smi

# PyTorch CUDA 호환성 확인
python3 -c "import torch; print(torch.cuda.is_available())"
```

#### 4. 모델 다운로드 실패
```bash
# 수동 모델 다운로드
cd /root/ws/src/risk_nav/models
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
```

#### 5. 토픽 연결 문제
```bash
# 토픽 확인
ros2 topic list
ros2 topic info /Camera/rgb

# 노드 확인
ros2 node list
ros2 node info /yolo_blip2_risk_node
```

### 성능 최적화 팁

#### GPU 사용량 최적화
```bash
# GPU 메모리 상태 확인
nvidia-smi

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

#### 속도 향상 설정
- **이미지 크기 조정**: 더 작은 입력 이미지 사용
- **모델 변경**: `yolo11n.pt` → `yolo11s.pt` (정확도 향상)
- **배치 처리**: 여러 이미지 동시 처리

### 로그 확인
```bash
# 노드 로그 확인
ros2 run rqt_console rqt_console

# 시스템 로그 확인
journalctl -u risk_nav

# Python 로그 확인
tail -f /var/log/syslog | grep yolo
```

## 📞 지원 및 문의

문제가 계속 발생하면 다음 정보와 함께 문의하세요:
- 운영체제 및 버전
- ROS2 버전
- Python 버전
- GPU 정보 (있는 경우)
- 에러 메시지 전체 내용

---

## 📄 라이센스
이 프로젝트는 MIT 라이센스 하에 있습니다.

## 🔗 관련 링크
- [YOLOv11 공식 문서](https://docs.ultralytics.com/)
- [BLIP2 모델 정보](https://huggingface.co/Salesforce/blip-image-captioning-base)
- [ROS2 Humble 문서](https://docs.ros.org/en/humble/) 