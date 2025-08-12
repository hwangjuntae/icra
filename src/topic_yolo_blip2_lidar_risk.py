#!/usr/bin/env python3
"""
YOLO + BLIP2 위험도 평가 ROS2 노드
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, LaserScan
import cv2
import numpy as np
import json
import time
import traceback
import os
from pathlib import Path
import hashlib
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import math

# cv_bridge import
try:
    from cv_bridge import CvBridge
    CV_BRIDGE_AVAILABLE = True
except ImportError:
    print("cv_bridge 패키지가 없습니다. ros-humble-cv-bridge를 설치해주세요.")
    CV_BRIDGE_AVAILABLE = False

# YOLO 사용을 위한 import
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("YOLO 패키지가 없습니다. ultralytics를 설치해주세요.")
    YOLO_AVAILABLE = False

# BLIP2 사용을 위한 import
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch
    from PIL import Image as PILImage
    BLIP_AVAILABLE = True
except ImportError:
    print("BLIP2 패키지가 없습니다. transformers를 설치해주세요.")
    BLIP_AVAILABLE = False

class YOLOBlip2RiskNode(Node):
    def __init__(self):
        super().__init__('yolo_blip2_risk_node')
        
        # cv_bridge 초기화
        if CV_BRIDGE_AVAILABLE:
            self.bridge = CvBridge()
        else:
            self.bridge = None
            self.get_logger().error("cv_bridge를 사용할 수 없습니다.")
        
        # Publisher 생성 (이미지 형태로 전송)
        self.risk_publisher = self.create_publisher(
            Image, 
            '/risk_assessment/image', 
            10
        )
        
        # 이미지 토픽 구독
        self.image_subscription = self.create_subscription(
            Image,
            '/Camera/rgb',
            self.image_callback,
            10
        )
        
        # LiDAR 구독 (토픽명 수정)
        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/Lidar/laser_scan',    # 실제 라이다 토픽
            self.scan_callback,
            10
        )
        self.latest_scan = None
        
        # 카메라 수평 시야각 (Field of View)
        self.camera_hfov = 90.0    # 카메라 스펙에 맞게 조정
        
        # 10-15Hz 제어를 위한 변수들
        self.target_fps = 12.0  # 10-15Hz 범위의 중간값으로 설정
        self.frame_interval = 1.0 / self.target_fps  # 0.083 seconds (83.3ms)
        self.last_process_time = 0.0
        self.latest_image = None
        
        # 해상도 설정 (처리 속도 향상을 위해)
        self.processing_scale = 0.8  # 80% 크기로 증가 (인식률 향상)
        self.min_processing_width = 640  # 최소 처리 너비 증가
        self.min_processing_height = 480  # 최소 처리 높이 증가
        
        # 성능 모니터링
        self.processing_times = []
        self.max_time_samples = 10
        
        # BLIP2 최적화 설정
        self.blip_batch_size = 4  # 배치 처리 크기
        self.blip_resolution = (224, 224)  # BLIP2 최적 해상도
        self.pending_blip_tasks = []
        
        # 스마트 캐싱
        self.risk_cache = OrderedDict()
        self.cache_max_size = 100
        self.cache_hit_count = 0
        self.cache_total_count = 0
        
        # 비동기 처리
        self.thread_executor = ThreadPoolExecutor(max_workers=2)
        self.blip_queue = []
        self.blip_results = {}
        
        # 타이머 생성
        self.processing_timer = self.create_timer(
            self.frame_interval,
            self.process_latest_image
        )
        
        # 모델 경로 설정
        self.yolo_model_path = Path("/root/ws/src/risk_nav/src/yolo11n.pt")
        
        # 모델 초기화
        self.initialize_models()
        
        # 위험도 카테고리 설정 (네비게이션 관점)
        self.risk_categories = {
            # 사람 관련 (가장 위험)
            "person": 90,
            "human": 90,
            
            # 동적 환경 (높은 위험)
            "door": 70,      # 갑자기 열릴 수 있음
            "window": 60,    # 사람이 나타날 수 있음
            "curtain": 50,   # 뒤에 사람이 있을 수 있음
            "blind": 50,     # 시야 차단
            "mirror": 40,    # 반사로 인한 혼란
            "glass": 65,     # 투명해서 놓칠 수 있음
            
            # 불안정한 물체 (중간 위험)
            "bottle": 45,    # 넘어질 수 있음
            "cup": 40,       # 넘어질 수 있음
            "vase": 60,      # 깨질 수 있음
            "plant": 35,     # 넘어질 수 있음
            "book": 30,      # 떨어질 수 있음
            "box": 40,       # 넘어질 수 있음
            "bag": 35,       # 넘어질 수 있음
            "chair": 50,     # 움직일 수 있음
            "table": 45,     # 움직일 수 있음
            "desk": 45,      # 움직일 수 있음
            
            # 전자기기 (중간 위험)
            "laptop": 40,    # 움직일 수 있음
            "monitor": 45,   # 넘어질 수 있음
            "keyboard": 25,  # 움직일 수 있음
            "mouse": 20,     # 움직일 수 있음
            "phone": 30,     # 움직일 수 있음
            "remote": 25,    # 움직일 수 있음
            
            # 안전장비 (낮은 위험)
            "fire extinguisher": 15,  # 고정되어 있음
            "exit": 10,      # 안전한 경로
            "sign": 5,       # 정보 제공
            
            # 기본값
            "default": {
                "base_risk": 30,
                "weight": "unknown",
                "temperature": "unknown",
                "stability": "unknown",
                "fragility": "unknown",
                "size": "unknown",
                "movement_speed": "unknown",
                "description": "알 수 없는 물체"
            }
        }
        
        # 객체 인식 최적화 설정
        self.confidence_threshold = 0.3  # 신뢰도 임계값 낮춤 (더 많은 객체 인식)
        self.nms_threshold = 0.4  # NMS 임계값
        self.max_detections = 20  # 최대 탐지 객체 수
        
        # 거리 기반 필터링 설정
        self.max_distance = 5.0  # 최대 거리 (미터)
        self.min_distance = 0.3  # 최소 거리 (미터)
        self.distance_weight = 0.7  # 거리 가중치
        
        # 위험도 카테고리 설정 (물체 기반 물리적 특성 고려)
        self.object_risk_profiles = {
            # 사람 관련 (가장 위험)
            "person": {
                "base_risk": 90,
                "weight": "variable",  # 사람 무게 (50-100kg)
                "temperature": "body_temp",  # 체온
                "stability": "dynamic",  # 움직일 수 있음
                "fragility": "high",  # 부상 위험
                "size": "large",
                "movement_speed": "variable",
                "description": "사람 - 가장 높은 위험도, 동적 움직임"
            },
            "human": {
                "base_risk": 90,
                "weight": "variable",
                "temperature": "body_temp",
                "stability": "dynamic",
                "fragility": "high",
                "size": "large",
                "movement_speed": "variable",
                "description": "사람 - 가장 높은 위험도, 동적 움직임"
            },
            
            # 뜨거운 물체 (높은 위험)
            "cup": {
                "base_risk": 40,
                "weight": "light",  # 0.2-0.5kg
                "temperature": "hot",  # 뜨거운 액체
                "stability": "unstable",  # 넘어질 수 있음
                "fragility": "medium",  # 깨질 수 있음
                "size": "small",
                "movement_speed": "static",
                "description": "뜨거운 컵 - 넘어지면 화상 위험"
            },
            "mug": {
                "base_risk": 45,
                "weight": "light",
                "temperature": "hot",
                "stability": "unstable",
                "fragility": "medium",
                "size": "small",
                "movement_speed": "static",
                "description": "뜨거운 머그컵 - 넘어지면 화상 위험"
            },
            "kettle": {
                "base_risk": 80,
                "weight": "medium",  # 1-2kg
                "temperature": "very_hot",  # 매우 뜨거움
                "stability": "stable",  # 상대적으로 안정적
                "fragility": "low",  # 금속으로 만들어짐
                "size": "medium",
                "movement_speed": "static",
                "description": "전기주전자 - 매우 뜨거운 물, 화상 위험"
            },
            
            # 무거운 물체 (중간-높은 위험)
            "chair": {
                "base_risk": 50,
                "weight": "heavy",  # 5-15kg
                "temperature": "room_temp",
                "stability": "movable",  # 움직일 수 있음
                "fragility": "low",  # 튼튼함
                "size": "large",
                "movement_speed": "slow",
                "description": "의자 - 무거워서 충돌 시 위험"
            },
            "table": {
                "base_risk": 45,
                "weight": "very_heavy",  # 20-50kg
                "temperature": "room_temp",
                "stability": "stable",  # 상대적으로 안정적
                "fragility": "low",
                "size": "very_large",
                "movement_speed": "static",
                "description": "테이블 - 매우 무거움, 충돌 시 위험"
            },
            "desk": {
                "base_risk": 45,
                "weight": "very_heavy",  # 30-80kg
                "temperature": "room_temp",
                "stability": "stable",
                "fragility": "low",
                "size": "very_large",
                "movement_speed": "static",
                "description": "책상 - 매우 무거움, 충돌 시 위험"
            },
            
            # 깨지기 쉬운 물체 (중간 위험)
            "vase": {
                "base_risk": 60,
                "weight": "medium",  # 1-3kg
                "temperature": "room_temp",
                "stability": "unstable",  # 넘어지기 쉬움
                "fragility": "very_high",  # 매우 깨지기 쉬움
                "size": "medium",
                "movement_speed": "static",
                "description": "화분 - 깨지기 쉬움, 파편 위험"
            },
            "bottle": {
                "base_risk": 45,
                "weight": "light",  # 0.5-1kg
                "temperature": "variable",  # 내용물에 따라
                "stability": "unstable",
                "fragility": "medium",
                "size": "small",
                "movement_speed": "static",
                "description": "병 - 넘어지기 쉬움, 깨질 수 있음"
            },
            "glass": {
                "base_risk": 70,
                "weight": "light",
                "temperature": "variable",
                "stability": "unstable",
                "fragility": "very_high",
                "size": "small",
                "movement_speed": "static",
                "description": "유리 - 매우 깨지기 쉬움, 파편 위험"
            },
            
            # 전자기기 (중간 위험)
            "laptop": {
                "base_risk": 40,
                "weight": "medium",  # 1-3kg
                "temperature": "warm",  # 사용 시 따뜻함
                "stability": "movable",
                "fragility": "high",  # 전자기기
                "size": "medium",
                "movement_speed": "static",
                "description": "노트북 - 전자기기, 손상 시 위험"
            },
            "monitor": {
                "base_risk": 45,
                "weight": "heavy",  # 3-8kg
                "temperature": "warm",
                "stability": "stable",
                "fragility": "high",
                "size": "large",
                "movement_speed": "static",
                "description": "모니터 - 무거운 전자기기"
            },
            
            # 작은 물체 (낮은 위험)
            "book": {
                "base_risk": 30,
                "weight": "light",  # 0.5-2kg
                "temperature": "room_temp",
                "stability": "stable",
                "fragility": "low",
                "size": "small",
                "movement_speed": "static",
                "description": "책 - 상대적으로 안전"
            },
            "phone": {
                "base_risk": 30,
                "weight": "very_light",  # 0.1-0.3kg
                "temperature": "warm",
                "stability": "movable",
                "fragility": "high",
                "size": "small",
                "movement_speed": "static",
                "description": "휴대폰 - 작지만 전자기기"
            },
            
            # 안전장비 (낮은 위험)
            "fire extinguisher": {
                "base_risk": 15,
                "weight": "heavy",  # 3-6kg
                "temperature": "room_temp",
                "stability": "stable",  # 고정되어 있음
                "fragility": "low",
                "size": "medium",
                "movement_speed": "static",
                "description": "소화기 - 안전장비, 고정됨"
            },
            
            # 기본값
            "default": {
                "base_risk": 30,
                "weight": "unknown",
                "temperature": "unknown",
                "stability": "unknown",
                "fragility": "unknown",
                "size": "unknown",
                "movement_speed": "unknown",
                "description": "알 수 없는 물체"
            }
        }
        
        # 무게 및 위험도 특성별 가중치
        self.weight_risk_weights = {
            "weight": {
                "very_light": 0.4,     # 0.5kg 이하
                "light": 0.7,          # 0.5-2kg
                "medium": 1.0,         # 2-8kg
                "heavy": 1.4,          # 8-20kg
                "very_heavy": 1.8      # 20kg 이상
            },
            "risk_type": {
                "static": 0.8,         # 정적 물체
                "dynamic": 1.3,        # 동적 물체
                "human": 1.6,          # 사람
                "sudden": 1.5,         # 돌발 상황
                "environmental": 0.9   # 환경 요소
            },
            "mobility": {
                "stationary": 0.8,     # 고정됨
                "movable": 1.2,        # 움직일 수 있음
                "mobile": 1.4,         # 이동 가능
                "unpredictable": 1.6   # 예측 불가능
            },
            "stability": {
                "stable": 0.8,         # 안정적
                "unstable": 1.4,       # 불안정
                "precarious": 1.3,     # 불안정
                "dynamic": 1.5         # 동적
            },
            "danger_level": {
                "safe": 0.5,           # 안전
                "low_risk": 0.8,       # 낮은 위험
                "medium_risk": 1.0,    # 보통 위험
                "high_risk": 1.4,      # 높은 위험
                "extremely_dangerous": 1.8  # 매우 위험
            },
            "size": {
                "very_small": 0.5,     # 매우 작음
                "small": 0.8,          # 작음
                "medium": 1.0,         # 보통
                "large": 1.3,          # 큼
                "very_large": 1.6      # 매우 큼
            }
        }
        
        # 거리 기반 위험도 계산 개선
        self.distance_risk_model = {
            "very_close": 2.0,    # 0-0.5m: 매우 위험
            "close": 1.5,          # 0.5-1m: 위험
            "medium": 1.2,         # 1-2m: 보통
            "far": 1.0,            # 2-3m: 안전
            "very_far": 0.8        # 3m 이상: 매우 안전
        }
        
        # 물리적 특성 분석 방법 설정
        self.analysis_method = 'blip2'  # 'blip2' 또는 'computer_vision'
        
        self.get_logger().info("YOLO + BLIP2 네비게이션 위험도 평가 노드 시작 (최적화 버전)")
        self.get_logger().info("구독 토픽: /Camera/rgb, /Lidar/laser_scan")
        self.get_logger().info("발행 토픽: /risk_assessment/image (sensor_msgs/Image)")
        self.get_logger().info(f"목표 처리 주파수: {self.target_fps}Hz ({self.frame_interval*1000:.1f}ms 간격)")
        self.get_logger().info(f"처리 해상도 스케일: {self.processing_scale:.1f} (최소: {self.min_processing_width}x{self.min_processing_height})")
        self.get_logger().info(f"YOLO 설정: conf={self.confidence_threshold}, nms={self.nms_threshold}, max_det={self.max_detections}")
        self.get_logger().info(f"거리 필터링: {self.min_distance}-{self.max_distance}m, 가중치={self.distance_weight}")
        self.get_logger().info(f"BLIP2 최적화: 배치크기={self.blip_batch_size}, 해상도={self.blip_resolution}, 캐시크기={self.cache_max_size}")
        self.get_logger().info(f"YOLO 모델 경로: {self.yolo_model_path}")
        self.get_logger().info("무게 및 위험도 특성 기반 위험도 평가 시스템")
        self.get_logger().info("BLIP2 통합 분석: 무게, 위험유형, 이동성, 위험도 레벨을 한 번에 분석")
        
    def initialize_models(self):
        """모델 초기화"""
        try:
            if YOLO_AVAILABLE:
                # YOLOv11 모델 로드
                self.get_logger().info(f"YOLOv11 모델 로드 중... 경로: {self.yolo_model_path}")
                
                # 모델이 존재하지 않으면 오류
                if not self.yolo_model_path.exists():
                    self.get_logger().error(f"YOLO 모델을 찾을 수 없습니다: {self.yolo_model_path}")
                    self.yolo_model = None
                else:
                    # YOLOv11 nano 모델 로드
                    self.yolo_model = YOLO(str(self.yolo_model_path))
                    self.get_logger().info(f"YOLOv11 모델 로드 완료: {self.yolo_model_path}")
            else:
                self.yolo_model = None
                
            if BLIP_AVAILABLE:
                self.get_logger().info("BLIP2 모델 로드 중...")
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                
                # GPU 가속 설정
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.blip_model.to(self.device)
                
                # FP16 사용으로 속도/메모리 최적화
                if self.device.type == 'cuda':
                    self.blip_model.half()
                    self.get_logger().info("GPU 가속 활성화 (FP16 모드)")
                
                # 모델 워밍업
                dummy_image = torch.randn(1, 3, 224, 224).to(self.device)
                if self.device.type == 'cuda':
                    dummy_image = dummy_image.half()
                with torch.no_grad():
                    _ = self.blip_model.generate(pixel_values=dummy_image, max_length=10)
                
                self.get_logger().info(f"BLIP2 모델 로드 완료 (디바이스: {self.device})")
            else:
                self.blip_processor = None
                self.blip_model = None
                self.device = torch.device('cpu')
                
        except Exception as e:
            self.get_logger().error(f"모델 초기화 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def process_latest_image(self):
        """10-15Hz로 최신 이미지를 처리하는 타이머 콜백"""
        if self.latest_image is None:
            return
            
        current_time = time.time()
        
        try:
            # 이미지 다운샘플링으로 처리 속도 향상
            processed_image, scale_info = self.downsample_image(self.latest_image)
            
            # 위험도 평가 수행 (다운샘플된 이미지로)
            risk_results = self.assess_risk(processed_image, scale_info)
            
            # 결과를 시각화한 이미지 생성 (원본 크기로)
            visualized_image = self.visualize_risk_results(self.latest_image, risk_results)
            
            # 시각화된 이미지를 ROS 이미지 메시지로 변환하여 퍼블리시
            if self.bridge is not None:
                try:
                    result_msg = self.bridge.cv2_to_imgmsg(visualized_image, "bgr8")
                    self.risk_publisher.publish(result_msg)
                    
                    # 처리 통계 계산 및 성능 모니터링
                    processing_time = (time.time() - current_time) * 1000  # ms
                    self.update_performance_stats(processing_time)
                    
                    avg_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else processing_time
                    actual_fps = 1000.0 / avg_time if avg_time > 0 else 0
                    
                    # 배치 처리 및 캐시 통계
                    cache_stats = self.get_cache_stats() if hasattr(self, 'cache_total_count') and self.cache_total_count > 0 else ""
                    blip_queue_size = len(self.blip_queue) if hasattr(self, 'blip_queue') else 0
                    
                    self.get_logger().info(f"BLIP2 네비게이션 위험도 평가: {len(risk_results.get('detections', []))}개 객체 | "
                                         f"처리: {processing_time:.1f}ms | FPS: {actual_fps:.1f} | "
                                         f"해상도: {processed_image.shape[1]}x{processed_image.shape[0]} | "
                                         f"BLIP 대기: {blip_queue_size} | {cache_stats}")
                    
                except Exception as bridge_error:
                    self.get_logger().error(f"이미지 변환 실패: {str(bridge_error)}")
            else:
                self.get_logger().error("cv_bridge를 사용할 수 없어 이미지 전송 실패")
            
        except Exception as e:
            self.get_logger().error(f"BLIP2 네비게이션 위험도 평가 이미지 처리 오류: {str(e)}")
            traceback.print_exc()
            
    def downsample_image(self, image):
        """이미지 다운샘플링으로 처리 속도 향상"""
        original_height, original_width = image.shape[:2]
        
        # 목표 크기 계산
        target_width = max(int(original_width * self.processing_scale), self.min_processing_width)
        target_height = max(int(original_height * self.processing_scale), self.min_processing_height)
        
        # 비율 유지하면서 크기 조정
        aspect_ratio = original_width / original_height
        if target_width / target_height > aspect_ratio:
            target_width = int(target_height * aspect_ratio)
        else:
            target_height = int(target_width / aspect_ratio)
        
        # 이미지 리사이즈
        resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        
        # 스케일 정보 저장
        scale_info = {
            'scale_x': original_width / target_width,
            'scale_y': original_height / target_height,
            'original_size': (original_width, original_height),
            'processed_size': (target_width, target_height)
        }
        
        return resized_image, scale_info
        
    def update_performance_stats(self, processing_time):
        """성능 통계 업데이트"""
        self.processing_times.append(processing_time)
        
        # 최대 샘플 수 유지
        if len(self.processing_times) > self.max_time_samples:
            self.processing_times.pop(0)
            
        # 성능이 너무 느리면 경고
        if len(self.processing_times) >= 5:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            expected_time = 1000.0 / self.target_fps  # ms
            
            if avg_time > expected_time * 1.5:  # 50% 이상 느리면
                self.get_logger().warn(f"처리 속도가 목표보다 느림: {avg_time:.1f}ms > {expected_time:.1f}ms")
                # 필요시 추가 최적화 수행
                if self.processing_scale > 0.5:  # 최소 50%까지만 줄임
                    self.processing_scale *= 0.9
                    self.get_logger().info(f"처리 스케일 조정: {self.processing_scale:.2f}")
                     
    def get_object_hash(self, obj_image):
        """객체 이미지의 해시값 생성 (캐싱용)"""
        # 이미지를 작은 크기로 리사이즈하여 해시 생성
        small_img = cv2.resize(obj_image, (32, 32))
        img_bytes = small_img.tobytes()
        return hashlib.md5(img_bytes).hexdigest()
        
    def get_cached_result(self, obj_hash):
        """캐시에서 결과 조회"""
        self.cache_total_count += 1
        if obj_hash in self.risk_cache:
            # LRU 업데이트
            result = self.risk_cache.pop(obj_hash)
            self.risk_cache[obj_hash] = result
            self.cache_hit_count += 1
            return result
        return None
        
    def cache_result(self, obj_hash, result):
        """결과를 캐시에 저장"""
        if len(self.risk_cache) >= self.cache_max_size:
            # 가장 오래된 항목 제거 (LRU)
            self.risk_cache.popitem(last=False)
        self.risk_cache[obj_hash] = result
        
    def get_cache_stats(self):
        """캐시 통계 반환"""
        if self.cache_total_count > 0:
            hit_rate = (self.cache_hit_count / self.cache_total_count) * 100
            return f"캐시 적중률: {hit_rate:.1f}% ({self.cache_hit_count}/{self.cache_total_count})"
        return "캐시 통계 없음"
        
    def optimize_image_for_blip(self, image):
        """BLIP2를 위한 이미지 최적화"""
        # BLIP2 최적 해상도로 리사이즈
        optimized = cv2.resize(image, self.blip_resolution, interpolation=cv2.INTER_LINEAR)
        return optimized
            
    def image_callback(self, msg):
        """이미지 콜백 함수 - 최신 이미지만 저장"""
        try:
            # cv_bridge가 있으면 사용, 없으면 수동 변환
            cv_image = None
            
            if self.bridge is not None:
                try:
                    cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                except Exception as bridge_error:
                    self.get_logger().warning(f"cv_bridge 변환 실패, 수동 변환 사용: {str(bridge_error)}")
                    cv_image = None
            
            # cv_bridge 실패하거나 없으면 수동 변환 사용
            if cv_image is None:
                cv_image = self.manual_image_conversion(msg)
                if cv_image is None:
                    self.get_logger().error("이미지 변환 실패")
                    return
            
            # 최신 이미지 업데이트 (타이머에서 처리됨)
            self.latest_image = cv_image
            
        except Exception as e:
            self.get_logger().error(f"이미지 콜백 오류: {str(e)}")
            traceback.print_exc()
            
    def manual_image_conversion(self, msg):
        """cv_bridge 대신 수동으로 이미지 변환"""
        try:
            # ROS 이미지 메시지에서 직접 numpy 배열로 변환
            if msg.encoding == 'bgr8':
                # BGR8 형태의 이미지
                np_arr = np.frombuffer(msg.data, dtype=np.uint8)
                cv_image = np_arr.reshape(msg.height, msg.width, 3)
            elif msg.encoding == 'rgb8':
                # RGB8 형태의 이미지를 BGR로 변환
                np_arr = np.frombuffer(msg.data, dtype=np.uint8)
                rgb_image = np_arr.reshape(msg.height, msg.width, 3)
                cv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'mono8':
                # 흑백 이미지를 컬러로 변환
                np_arr = np.frombuffer(msg.data, dtype=np.uint8)
                gray_image = np_arr.reshape(msg.height, msg.width)
                cv_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
            else:
                self.get_logger().error(f"지원되지 않는 이미지 인코딩: {msg.encoding}")
                return None
                
            self.get_logger().info(f"수동 이미지 변환 성공: {msg.encoding} -> BGR")
            return cv_image
            
        except Exception as e:
            self.get_logger().error(f"수동 이미지 변환 실패: {str(e)}")
            return None
            
    def visualize_risk_results(self, image, risk_results):
        """위험도 평가 결과를 시각화한 이미지 생성"""
        # 원본 이미지 복사
        vis_image = image.copy()
        
        # 위험도 레벨별 색상 매핑 (BGR 형태)
        color_map = {
            "low": (0, 255, 0),      # 녹색
            "medium": (0, 255, 255), # 노란색
            "high": (0, 0, 255)      # 빨간색
        }
        
        # 각 탐지된 객체에 대해 바운딩 박스와 정보 표시
        for detection in risk_results.get('detections', []):
            # 바운딩 박스 좌표
            bbox = detection['bbox']
            x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
            
            # 위험도 레벨에 따른 색상 선택
            risk_level = detection['risk_level']
            color = color_map.get(risk_level, (0, 255, 0))
            
            # 바운딩 박스 그리기
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # 라벨 텍스트 생성
            class_name = detection['class_name']
            confidence = detection['confidence']
            risk_score = detection['risk_score']
            distance = detection.get('distance', 'N/A')
            
            # 무게 및 위험도 특성 정보 생성
            try:
                # BLIP2로 무게와 위험도 특성 분석
                properties = self.analyze_weight_and_risk_properties(image, class_name, detection['bbox'])
                
                weight_info = f"W:{properties['weight']}"
                risk_type_info = f"R:{properties['risk_type']}"
                mobility_info = f"M:{properties['mobility']}"
                danger_info = f"D:{properties['danger_level']}"
            except Exception as e:
                # 분석 실패 시 기본 정보 사용
                profile = self.object_risk_profiles.get(class_name, self.object_risk_profiles["default"])
                weight_info = f"W:{profile['weight']}" if profile['weight'] != 'unknown' else ""
                risk_type_info = ""
                mobility_info = ""
                danger_info = ""
            
            label = f"{class_name} {confidence:.2f}"
            risk_text = f"Risk: {risk_score:.1f} ({risk_level})"
            distance_text = f"Dist: {distance}m" if isinstance(distance, (int, float)) else f"Dist: {distance}"
            physics_text = f"{weight_info} {risk_type_info} {mobility_info} {danger_info}".strip()
            
            # 텍스트 배경 박스 크기 계산
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            (risk_w, risk_h), _ = cv2.getTextSize(risk_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            (dist_w, dist_h), _ = cv2.getTextSize(distance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            (physics_w, physics_h), _ = cv2.getTextSize(physics_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            
            # 텍스트 배경 박스 그리기
            max_text_w = max(label_w, risk_w, dist_w, physics_w)
            cv2.rectangle(vis_image, (x1, y1 - label_h - risk_h - dist_h - physics_h - 20), 
                         (x1 + max_text_w + 10, y1), color, -1)
            
            # 텍스트 그리기
            cv2.putText(vis_image, label, (x1 + 5, y1 - dist_h - risk_h - physics_h - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(vis_image, risk_text, (x1 + 5, y1 - dist_h - physics_h - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            cv2.putText(vis_image, distance_text, (x1 + 5, y1 - physics_h - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            cv2.putText(vis_image, physics_text, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        
        # 전체 위험도 정보 표시
        overall_risk = risk_results.get('overall_risk_score', 0)
        overall_level = risk_results.get('risk_level', 'low')
        detection_count = len(risk_results.get('detections', []))
        
        # 상단에 전체 정보 표시
        info_text = f"Objects: {detection_count} | Navigation Risk: {overall_risk:.1f} ({overall_level})"
        
        # 정보 배경 박스
        (info_w, info_h), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(vis_image, (10, 10), (20 + info_w, 30 + info_h), (0, 0, 0), -1)
        
        # 전체 위험도 색상
        overall_color = color_map.get(overall_level, (0, 255, 0))
        cv2.putText(vis_image, info_text, (15, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, overall_color, 2)
        
        # 장면 설명 표시 (하단)
        scene_desc = risk_results.get('scene_description', '')
        if scene_desc and len(scene_desc) > 0:
            # 긴 텍스트를 적절히 자르기
            if len(scene_desc) > 60:
                scene_desc = scene_desc[:60] + "..."
            
            height, width = vis_image.shape[:2]
            (scene_w, scene_h), _ = cv2.getTextSize(scene_desc, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # 하단에 장면 설명 배경 박스
            cv2.rectangle(vis_image, (10, height - scene_h - 20), 
                         (20 + scene_w, height - 10), (0, 0, 0), -1)
            
            # 장면 설명 텍스트
            cv2.putText(vis_image, f"Navigation Safety: {scene_desc}", (15, height - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_image
            
    def assess_risk(self, image, scale_info=None):
        """위험도 평가 수행 (LiDAR 융합 포함)"""
        results = {
            "timestamp": time.time(),
            "detections": [],
            "overall_risk_score": 0.0,
            "risk_level": "low",
            "scene_description": "네비게이션 안전성 분석 중..."
        }
        try:
            if self.yolo_model is not None:
                # YOLO 추론 수행 (최적화된 설정)
                yolo_results = self.yolo_model(
                    image,
                    conf=self.confidence_threshold,
                    iou=self.nms_threshold,
                    max_det=self.max_detections,
                    verbose=False
                )
                
                if len(yolo_results) > 0 and yolo_results[0].boxes is not None:
                    detections = []
                    total_risk = 0.0
                    detection_count = 0
                    
                    for detection in yolo_results[0].boxes:
                        x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
                        confidence = float(detection.conf[0].cpu().numpy())
                        class_id = int(detection.cls[0].cpu().numpy())
                        class_name = self.yolo_model.names[class_id]
                        
                        # 스케일 정보 적용
                        if scale_info is not None:
                            x1 *= scale_info['scale_x']
                            y1 *= scale_info['scale_y']
                            x2 *= scale_info['scale_x']
                            y2 *= scale_info['scale_y']
                        
                        # 거리 정보 계산
                        distance = self.calculate_object_distance(x1, y1, x2, y2, scale_info)
                        
                        # 거리 기반 필터링
                        if distance is not None and (distance < self.min_distance or distance > self.max_distance):
                            continue
                        
                        # 동적 물리적 특성을 고려한 위험도 점수 계산
                        bbox = {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
                        risk_score = self.calculate_risk_score_with_distance(class_name, confidence, distance, image, bbox)
                        risk_level = self.get_risk_level(risk_score)
                        
                        detection_result = {
                            "class_name": class_name,
                            "confidence": confidence,
                            "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                            "risk_score": float(risk_score),
                            "risk_level": risk_level,
                            "description": f"{class_name} 객체",
                            "distance": distance,
                            "lidar_overlap": distance is not None
                        }
                        
                        detections.append(detection_result)
                        total_risk += risk_score
                        detection_count += 1
                        
                        # BLIP2 분석을 위한 객체 이미지 추출 및 큐에 추가
                        if self.blip_model is not None:
                            try:
                                # 객체 이미지 추출
                                obj_image = image[int(y1):int(y2), int(x1):int(x2)]
                                if obj_image.size > 0:  # 유효한 이미지인지 확인
                                    self.blip_queue.append({
                                        'image': obj_image,
                                        'class_name': class_name,
                                        'bbox_idx': len(detections) - 1  # 현재 인덱스
                                    })
                            except Exception as e:
                                self.get_logger().debug(f"객체 이미지 추출 실패: {str(e)}")
                    
                    # 거리 기반 정렬 (가까운 객체 우선)
                    detections.sort(key=lambda d: d.get('distance', float('inf')) if d.get('distance') is not None else float('inf'))
                    
                    # 최대 탐지 수 제한
                    detections = detections[:self.max_detections]
                    
                    results["detections"] = detections
                    
                    if detection_count > 0:
                        results["overall_risk_score"] = total_risk / detection_count
                        results["risk_level"] = self.get_risk_level(results["overall_risk_score"])
                        
                        self.get_logger().info(f"YOLO 탐지 + BLIP2 네비게이션 위험도 평가: {detection_count}개 객체 (거리순 정렬)")
            
            # BLIP2 분석 수행 및 결과 반영
            if self.blip_model is not None and self.blip_queue:
                # 동기적으로 BLIP2 처리 (실시간 반영을 위해)
                self.process_blip_batch_sync(self.blip_queue.copy(), results)
                self.blip_queue.clear()
                
                # BLIP2 결과로 최종 위험도 재계산
                if results["detections"]:
                    total_blip_risk = 0.0
                    valid_detections = 0
                    
                    for detection in results["detections"]:
                        if 'risk_score' in detection:
                            total_blip_risk += detection['risk_score']
                            valid_detections += 1
                    
                    if valid_detections > 0:
                        results["overall_risk_score"] = total_blip_risk / valid_detections
                        results["risk_level"] = self.get_risk_level(results["overall_risk_score"])
                        self.get_logger().info(f"BLIP2 위험도 재계산: {results['overall_risk_score']:.1f} ({results['risk_level']})")
                
                # 전체 장면 설명
                results["scene_description"] = self.get_scene_description(image)
                
        except Exception as e:
            self.get_logger().error(f"BLIP2 네비게이션 위험도 평가 오류: {str(e)}")
            traceback.print_exc()
            
        return results
        
    def calculate_object_distance(self, x1, y1, x2, y2, scale_info=None):
        """객체의 거리 계산 (LiDAR 융합)"""
        if self.latest_scan is None or scale_info is None:
            return None
            
        try:
            # 객체 중심점 계산
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            
            # 이미지 중심에서의 상대적 위치
            img_width = scale_info['original_size'][0]
            img_height = scale_info['original_size'][1]
            
            # 정규화된 좌표 (-1 to 1)
            rel_x = (center_x / img_width) * 2.0 - 1.0  # -1 (왼쪽) to 1 (오른쪽)
            
            # 카메라 시야각을 라이다 각도로 변환
            angle_rad = rel_x * math.radians(self.camera_hfov / 2.0)
            
            # 라이다 인덱스 계산
            if self.latest_scan.angle_increment != 0:
                idx = int((angle_rad - self.latest_scan.angle_min) / self.latest_scan.angle_increment)
                
                if 0 <= idx < len(self.latest_scan.ranges):
                    distance = self.latest_scan.ranges[idx]
                    
                    # 유효한 거리인지 확인
                    if not math.isnan(distance) and distance > 0 and distance < self.latest_scan.range_max:
                        return distance
                        
        except Exception as e:
            self.get_logger().debug(f"거리 계산 오류: {str(e)}")
            
        return None
        
    def calculate_risk_score_with_distance(self, class_name, confidence, distance, image=None, bbox=None):
        """동적 물리적 특성을 고려한 거리 위험도 점수 계산"""
        # 기본 위험도 점수 (동적 물리적 특성 포함)
        base_score = self.calculate_risk_score(class_name, confidence, image, bbox)
        
        # 거리 기반 위험도 계산 (개선된 모델)
        if distance is not None:
            # 거리 구간별 위험도 가중치
            if distance <= 0.5:
                distance_factor = self.distance_risk_model["very_close"]
            elif distance <= 1.0:
                distance_factor = self.distance_risk_model["close"]
            elif distance <= 2.0:
                distance_factor = self.distance_risk_model["medium"]
            elif distance <= 3.0:
                distance_factor = self.distance_risk_model["far"]
            else:
                distance_factor = self.distance_risk_model["very_far"]
            
            # 무게 및 위험도 특성에 따른 거리 가중치 조정
            if image is not None and bbox is not None:
                try:
                    properties = self.analyze_weight_and_risk_properties(image, class_name, bbox)
                    
                    if properties["risk_type"] == "human":
                        # 사람은 가까울 때 매우 위험
                        if distance <= 2.0:
                            distance_factor *= 1.6
                    elif properties["risk_type"] == "sudden":
                        # 돌발 상황은 가까울 때 매우 위험
                        if distance <= 1.5:
                            distance_factor *= 1.5
                    elif properties["mobility"] == "unpredictable":
                        # 예측 불가능한 물체는 가까울 때 위험
                        if distance <= 1.0:
                            distance_factor *= 1.4
                    elif properties["stability"] == "unstable":
                        # 불안정한 물체는 가까울 때 위험
                        if distance <= 1.0:
                            distance_factor *= 1.3
                    elif properties["weight"] in ["heavy", "very_heavy"]:
                        # 무거운 물체는 충돌 시 위험
                        if distance <= 2.0:
                            distance_factor *= 1.2
                    elif properties["danger_level"] == "extremely_dangerous":
                        # 매우 위험한 물체는 가까울 때 위험
                        if distance <= 1.5:
                            distance_factor *= 1.4
                except Exception as e:
                    self.get_logger().error(f"거리 가중치 조정 실패: {str(e)}")
            else:
                # 기본 프로파일 사용
                profile = self.object_risk_profiles.get(class_name, self.object_risk_profiles["default"])
                if profile["stability"] == "unstable":
                    if distance <= 1.0:
                        distance_factor *= 1.3
                elif profile["weight"] in ["heavy", "very_heavy"]:
                    if distance <= 2.0:
                        distance_factor *= 1.2
            
            base_score *= distance_factor
        
        # 물체별 특수 위험도 조정
        if properties.get("risk_type") == "human" and distance is not None and distance <= 2.0:
            # 사람은 동적 움직임으로 인한 위험
            base_score *= 1.6
        elif properties.get("risk_type") == "sudden" and distance is not None and distance <= 1.5:
            # 돌발 상황은 예측 불가능한 위험
            base_score *= 1.5
        elif properties.get("mobility") == "unpredictable" and distance is not None and distance <= 1.0:
            # 예측 불가능한 물체는 돌발 위험
            base_score *= 1.4
        elif properties.get("weight") in ["heavy", "very_heavy"] and distance is not None and distance <= 2.0:
            # 무거운 물체는 충돌 시 심각한 위험
            base_score *= 1.3
        elif properties.get("danger_level") == "extremely_dangerous" and distance is not None and distance <= 1.5:
            # 매우 위험한 물체는 즉시 회피 필요
            base_score *= 1.7
        
        return max(0, min(100, base_score))
        
    def process_blip_batch(self, batch_objects, results):
        """배치로 BLIP2 처리 (비동기)"""
        try:
            # 캐시 확인 및 새로 처리할 객체 분리
            cache_hits = 0
            to_process = []
            
            for obj_info in batch_objects:
                obj_hash = self.get_object_hash(obj_info['image'])
                cached_result = self.get_cached_result(obj_hash)
                
                if cached_result is not None:
                    # 캐시 히트
                    cache_hits += 1
                    bbox_idx = obj_info['bbox_idx']
                    if bbox_idx < len(results['detections']):
                        results['detections'][bbox_idx]['description'] = cached_result
                        results['detections'][bbox_idx]['risk_score'] = self.calculate_blip_risk_score(cached_result)
                        results['detections'][bbox_idx]['risk_level'] = self.get_risk_level(results['detections'][bbox_idx]['risk_score'])
                else:
                    to_process.append((obj_info, obj_hash))
            
            if cache_hits > 0:
                self.get_logger().info(f"캐시 히트: {cache_hits}개 객체 | {self.get_cache_stats()}")
            
            # 새로 처리할 객체들을 배치로 처리
            if to_process:
                batch_size = min(len(to_process), self.blip_batch_size)
                for i in range(0, len(to_process), batch_size):
                    batch = to_process[i:i + batch_size]
                    self.process_object_batch(batch, results)
                    
        except Exception as e:
            self.get_logger().error(f"배치 처리 오류: {str(e)}")
            
    def process_object_batch(self, batch, results):
        """객체 배치 BLIP2 처리"""
        try:
            if not batch or self.blip_model is None:
                return
                
            # 배치 이미지 준비
            batch_images = []
            batch_prompts = []
            
            for obj_info, obj_hash in batch:
                # 해상도 최적화
                optimized_img = self.optimize_image_for_blip(obj_info['image'])
                rgb_img = cv2.cvtColor(optimized_img, cv2.COLOR_BGR2RGB)
                pil_img = PILImage.fromarray(rgb_img)
                
                batch_images.append(pil_img)
                # 네비게이션 위험도 평가 특화 프롬프트
                prompt = f"Is this {obj_info['class_name']} object likely to move, fall, or cause navigation issues? Assess the navigation risk level."
                batch_prompts.append(prompt)
            
            # 배치 처리
            inputs = self.blip_processor(batch_images, batch_prompts, return_tensors="pt", padding=True).to(self.device)
            
            # FP16 처리
            if self.device.type == 'cuda':
                for key in inputs:
                    if inputs[key].dtype == torch.float32:
                        inputs[key] = inputs[key].half()
            
            with torch.no_grad():
                outputs = self.blip_model.generate(**inputs, max_length=100, do_sample=False)
            
            # 결과 처리
            descriptions = [self.blip_processor.decode(output, skip_special_tokens=True) for output in outputs]
            
            for i, ((obj_info, obj_hash), description) in enumerate(zip(batch, descriptions)):
                # 결과 캐시
                self.cache_result(obj_hash, description)
                
                # 결과 업데이트
                bbox_idx = obj_info['bbox_idx']
                if bbox_idx < len(results['detections']):
                    results['detections'][bbox_idx]['description'] = description
                    results['detections'][bbox_idx]['risk_score'] = self.calculate_blip_risk_score(description)
                    results['detections'][bbox_idx]['risk_level'] = self.get_risk_level(results['detections'][bbox_idx]['risk_score'])
                    
        except Exception as e:
            self.get_logger().error(f"객체 배치 처리 오류: {str(e)}")
            
    def process_blip_batch_sync(self, batch_objects, results):
        """동기적으로 BLIP2 배치 처리 (실시간 반영)"""
        try:
            if not batch_objects or self.blip_model is None:
                return
                
            # 캐시 확인 및 새로 처리할 객체 분리
            cache_hits = 0
            to_process = []
            
            for obj_info in batch_objects:
                obj_hash = self.get_object_hash(obj_info['image'])
                cached_result = self.get_cached_result(obj_hash)
                
                if cached_result is not None:
                    # 캐시 히트
                    cache_hits += 1
                    bbox_idx = obj_info['bbox_idx']
                    if bbox_idx < len(results['detections']):
                        results['detections'][bbox_idx]['description'] = cached_result
                        results['detections'][bbox_idx]['risk_score'] = self.calculate_blip_risk_score(cached_result)
                        results['detections'][bbox_idx]['risk_level'] = self.get_risk_level(results['detections'][bbox_idx]['risk_score'])
                else:
                    to_process.append((obj_info, obj_hash))
            
            if cache_hits > 0:
                self.get_logger().info(f"BLIP2 캐시 히트: {cache_hits}개 객체")
            
            # 새로 처리할 객체들을 배치로 처리
            if to_process:
                batch_size = min(len(to_process), self.blip_batch_size)
                for i in range(0, len(to_process), batch_size):
                    batch = to_process[i:i + batch_size]
                    self.process_object_batch_sync(batch, results)
                    
        except Exception as e:
            self.get_logger().error(f"동기 BLIP2 배치 처리 오류: {str(e)}")
            
    def process_object_batch_sync(self, batch, results):
        """동기적으로 객체 배치 BLIP2 처리"""
        try:
            if not batch or self.blip_model is None:
                return
                
            # 배치 이미지 준비
            batch_images = []
            batch_prompts = []
            
            for obj_info, obj_hash in batch:
                # 해상도 최적화
                optimized_img = self.optimize_image_for_blip(obj_info['image'])
                rgb_img = cv2.cvtColor(optimized_img, cv2.COLOR_BGR2RGB)
                pil_img = PILImage.fromarray(rgb_img)
                
                batch_images.append(pil_img)
                # 네비게이션 위험도 평가 특화 프롬프트
                prompt = f"Is this {obj_info['class_name']} object likely to move, fall, or cause navigation issues? Assess the navigation risk level."
                batch_prompts.append(prompt)
            
            # 배치 처리
            inputs = self.blip_processor(batch_images, batch_prompts, return_tensors="pt", padding=True).to(self.device)
            
            # FP16 처리
            if self.device.type == 'cuda':
                for key in inputs:
                    if inputs[key].dtype == torch.float32:
                        inputs[key] = inputs[key].half()
            
            with torch.no_grad():
                outputs = self.blip_model.generate(**inputs, max_length=100, do_sample=False)
            
            # 결과 처리
            descriptions = [self.blip_processor.decode(output, skip_special_tokens=True) for output in outputs]
            
            for i, ((obj_info, obj_hash), description) in enumerate(zip(batch, descriptions)):
                # 결과 캐시
                self.cache_result(obj_hash, description)
                
                # 결과 업데이트
                bbox_idx = obj_info['bbox_idx']
                if bbox_idx < len(results['detections']):
                    results['detections'][bbox_idx]['description'] = description
                    results['detections'][bbox_idx]['risk_score'] = self.calculate_blip_risk_score(description)
                    results['detections'][bbox_idx]['risk_level'] = self.get_risk_level(results['detections'][bbox_idx]['risk_score'])
                    
        except Exception as e:
            self.get_logger().error(f"동기 객체 배치 처리 오류: {str(e)}")
            
    def calculate_blip_risk_score(self, description):
        """BLIP2 설명에서 네비게이션 위험도 점수 계산"""
        description_lower = description.lower()
        
        # 네비게이션 위험 키워드 점수
        navigation_danger_keywords = {
            'move': 60, 'moving': 65, 'unstable': 70, 'wobbly': 75, 'shaky': 70,
            'fall': 80, 'falling': 85, 'tipping': 75, 'unbalanced': 70,
            'block': 50, 'blocking': 55, 'obstacle': 60, 'blocked': 55,
            'dynamic': 65, 'changing': 60, 'unpredictable': 80,
            'crowded': 70, 'busy': 60, 'traffic': 65,
            'narrow': 50, 'tight': 55, 'confined': 60,
            'slippery': 75, 'wet': 70, 'smooth': 45,
            'hidden': 65, 'concealed': 60, 'behind': 50,
            'sudden': 80, 'unexpected': 85, 'surprise': 75,
            'collision': 90, 'crash': 95, 'impact': 85
        }
        
        # 안전 키워드 점수 (위험도 감소)
        navigation_safe_keywords = {
            'stable': -30, 'fixed': -25, 'secure': -20, 'stationary': -35,
            'safe': -20, 'clear': -15, 'open': -10, 'wide': -15,
            'static': -30, 'immobile': -35, 'anchored': -40,
            'predictable': -25, 'consistent': -20, 'reliable': -15
        }
        
        base_score = 30  # 기본 점수 (네비게이션 관점)
        
        for keyword, score in navigation_danger_keywords.items():
            if keyword in description_lower:
                base_score += score * 0.6  # 가중치 적용
                
        for keyword, score in navigation_safe_keywords.items():
            if keyword in description_lower:
                base_score += score * 0.4  # 가중치 적용
                
        return max(0, min(100, base_score))
        
    def calculate_risk_score(self, class_name, confidence, image=None, bbox=None):
        """무게와 위험도 특성을 고려한 네비게이션 위험도 점수 계산"""
        # 기본 위험도 점수
        base_score = 30  # 기본값
        
        # 무게와 위험도 특성 분석 (이미지가 제공된 경우)
        if image is not None and bbox is not None:
            try:
                # BLIP2로 무게와 위험도 특성 분석
                properties = self.analyze_weight_and_risk_properties(image, class_name, bbox)
                
                # 무게 및 위험도 특성별 가중치 적용
                weight_factor = self.weight_risk_weights["weight"].get(properties["weight"], 1.0)
                risk_type_factor = self.weight_risk_weights["risk_type"].get(properties["risk_type"], 1.0)
                mobility_factor = self.weight_risk_weights["mobility"].get(properties["mobility"], 1.0)
                stability_factor = self.weight_risk_weights["stability"].get(properties["stability"], 1.0)
                danger_factor = self.weight_risk_weights["danger_level"].get(properties["danger_level"], 1.0)
                size_factor = self.weight_risk_weights["size"].get(properties["size"], 1.0)
                
                # 기본 위험도 계산 (물체 종류별)
                if class_name in ["person", "human"]:
                    base_score = 90
                elif properties["risk_type"] == "human":
                    base_score = 85
                elif properties["risk_type"] == "sudden":
                    base_score = 75
                elif properties["risk_type"] == "dynamic":
                    base_score = 65
                elif properties["danger_level"] == "extremely_dangerous":
                    base_score = 80
                elif properties["danger_level"] == "high_risk":
                    base_score = 60
                elif properties["mobility"] == "unpredictable":
                    base_score = 70
                else:
                    base_score = 30
                
                # 무게 및 위험도 특성 가중치 적용
                base_score *= weight_factor * risk_type_factor * mobility_factor * stability_factor * danger_factor * size_factor
                
                self.get_logger().debug(f"무게/위험도 분석: {class_name} - 무게:{properties['weight']}, 위험유형:{properties['risk_type']}, 이동성:{properties['mobility']}, 위험도:{base_score:.1f}")
                
            except Exception as e:
                self.get_logger().error(f"무게/위험도 특성 분석 실패: {str(e)}")
                # 실패 시 기본 위험도 사용
                base_score = 30
        else:
            # 이미지가 없는 경우 기본 프로파일 사용
            profile = self.object_risk_profiles.get(class_name, self.object_risk_profiles["default"])
            base_score = profile["base_risk"]
            
            # 기본 가중치 적용
            if profile["weight"] != "unknown":
                weight_factor = self.weight_risk_weights["weight"].get(profile["weight"], 1.0)
                base_score *= weight_factor
        
        # 신뢰도 가중치 적용
        confidence_weight = confidence * 15
        
        # 네비게이션 관점에서 특정 객체별 위험도 조정
        if class_name in ["person", "human"]:
            adjustment = 20  # 사람은 가장 위험
        elif class_name in ["door", "window", "glass"]:
            adjustment = 15  # 동적 환경 요소
        elif class_name in ["bottle", "cup", "vase", "plant"]:
            adjustment = 10  # 넘어질 수 있는 물체
        elif class_name in ["fire extinguisher", "exit", "sign"]:
            adjustment = -10  # 안전장비는 위험도 감소
        else:
            adjustment = 0
            
        final_score = base_score + confidence_weight + adjustment
        return max(0, min(100, final_score))
        
    def get_risk_level(self, score):
        """위험도 레벨 결정"""
        if score < 30:
            return "low"
        elif score < 70:
            return "medium"
        else:
            return "high"
            
    def get_object_description(self, image, x1, y1, x2, y2, class_name):
        """객체 설명 생성 (레거시 - 배치 처리로 대체됨)"""
        return f"{class_name} 객체 (배치 처리 중)"
            
    def get_scene_description(self, image):
        """전체 장면 설명 생성 (네비게이션 위험도 관점)"""
        if self.blip_model is None:
            return "장면 설명을 생성할 수 없습니다."
            
        try:
            # 해상도 최적화
            optimized_img = self.optimize_image_for_blip(image)
            rgb_image = cv2.cvtColor(optimized_img, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb_image)
            
            # 네비게이션 안전성 특화 프롬프트
            prompt = "Describe the navigation safety of this scene. Are there moving objects, unstable items, or potential collision risks for a robot?"
            
            # BLIP2로 장면 설명 생성
            inputs = self.blip_processor(pil_image, prompt, return_tensors="pt").to(self.device)
            
            # FP16 처리
            if self.device.type == 'cuda':
                for key in inputs:
                    if inputs[key].dtype == torch.float32:
                        inputs[key] = inputs[key].half()
            
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=80, do_sample=False)
            description = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            return description
            
        except Exception as e:
            self.get_logger().error(f"장면 설명 생성 실패: {str(e)}")
            return "장면 설명을 생성할 수 없습니다."

    def analyze_object_physical_properties(self, image, class_name, bbox):
        """BLIP2를 사용하여 물체의 물리적 특성을 동적으로 분석"""
        try:
            # 바운딩 박스 영역 추출
            x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
            object_image = image[y1:y2, x1:x2]
            
            if object_image.size == 0:
                return self.get_default_properties()
            
            # BLIP2로 물리적 특성 분석
            properties = {
                'weight': self.analyze_weight(object_image, class_name),
                'temperature': self.analyze_temperature(object_image, class_name),
                'stability': self.analyze_stability(object_image, class_name),
                'fragility': self.analyze_fragility(object_image, class_name),
                'size': self.analyze_size(object_image, class_name),
                'movement_speed': self.analyze_movement_speed(object_image, class_name)
            }
            
            return properties
            
        except Exception as e:
            self.get_logger().error(f"물리적 특성 분석 오류: {str(e)}")
            return self.get_default_properties()
    
    def analyze_weight(self, image, class_name):
        """BLIP2로 무게 분석"""
        try:
            prompt = f"What is the weight of this {class_name}? Is it very light, light, medium, heavy, or very heavy?"
            result = self.blip_analysis(image, prompt)
            
            # 결과에서 무게 정보 추출
            result_lower = result.lower()
            if 'very light' in result_lower or 'lightweight' in result_lower:
                return 'very_light'
            elif 'light' in result_lower:
                return 'light'
            elif 'medium' in result_lower or 'moderate' in result_lower:
                return 'medium'
            elif 'heavy' in result_lower:
                return 'heavy'
            elif 'very heavy' in result_lower or 'extremely heavy' in result_lower:
                return 'very_heavy'
            else:
                return 'medium'  # 기본값
                
        except Exception as e:
            return 'medium'
    
    def analyze_temperature(self, image, class_name):
        """BLIP2로 온도 분석"""
        try:
            prompt = f"What is the temperature of this {class_name}? Is it cold, room temperature, warm, hot, or very hot?"
            result = self.blip_analysis(image, prompt)
            
            result_lower = result.lower()
            if 'cold' in result_lower or 'cool' in result_lower:
                return 'cold'
            elif 'room temperature' in result_lower or 'normal' in result_lower:
                return 'room_temp'
            elif 'warm' in result_lower:
                return 'warm'
            elif 'hot' in result_lower:
                return 'hot'
            elif 'very hot' in result_lower or 'extremely hot' in result_lower:
                return 'very_hot'
            else:
                return 'room_temp'
                
        except Exception as e:
            return 'room_temp'
    
    def analyze_stability(self, image, class_name):
        """BLIP2로 안정성 분석"""
        try:
            prompt = f"Is this {class_name} stable, movable, unstable, or dynamic? Can it fall or move easily?"
            result = self.blip_analysis(image, prompt)
            
            result_lower = result.lower()
            if 'stable' in result_lower or 'fixed' in result_lower:
                return 'stable'
            elif 'movable' in result_lower or 'can move' in result_lower:
                return 'movable'
            elif 'unstable' in result_lower or 'can fall' in result_lower:
                return 'unstable'
            elif 'dynamic' in result_lower or 'moving' in result_lower:
                return 'dynamic'
            else:
                return 'stable'
                
        except Exception as e:
            return 'stable'
    
    def analyze_fragility(self, image, class_name):
        """BLIP2로 깨지기 쉬움 분석"""
        try:
            prompt = f"Is this {class_name} fragile, breakable, or sturdy? Can it break easily?"
            result = self.blip_analysis(image, prompt)
            
            result_lower = result.lower()
            if 'fragile' in result_lower or 'breakable' in result_lower or 'easily break' in result_lower:
                return 'very_high'
            elif 'somewhat fragile' in result_lower or 'can break' in result_lower:
                return 'high'
            elif 'moderate' in result_lower or 'medium' in result_lower:
                return 'medium'
            elif 'sturdy' in result_lower or 'strong' in result_lower:
                return 'low'
            else:
                return 'medium'
                
        except Exception as e:
            return 'medium'
    
    def analyze_size(self, image, class_name):
        """이미지 크기 기반으로 크기 분석"""
        try:
            height, width = image.shape[:2]
            area = height * width
            
            # 상대적 크기 계산
            if area < 1000:  # 매우 작음
                return 'very_small'
            elif area < 5000:  # 작음
                return 'small'
            elif area < 20000:  # 보통
                return 'medium'
            elif area < 50000:  # 큼
                return 'large'
            else:  # 매우 큼
                return 'very_large'
                
        except Exception as e:
            return 'medium'
    
    def analyze_movement_speed(self, image, class_name):
        """BLIP2로 움직임 속도 분석"""
        try:
            prompt = f"Can this {class_name} move? Is it static, slow moving, fast moving, or variable speed?"
            result = self.blip_analysis(image, prompt)
            
            result_lower = result.lower()
            if 'static' in result_lower or 'stationary' in result_lower:
                return 'static'
            elif 'slow' in result_lower:
                return 'slow'
            elif 'fast' in result_lower or 'quick' in result_lower:
                return 'fast'
            elif 'variable' in result_lower or 'dynamic' in result_lower:
                return 'variable'
            else:
                return 'static'
                
        except Exception as e:
            return 'static'
    
    def blip_analysis(self, image, prompt):
        """BLIP2로 단일 이미지 분석"""
        try:
            if self.blip_model is None:
                return "unknown"
            
            # 이미지 최적화
            optimized_img = self.optimize_image_for_blip(image)
            rgb_img = cv2.cvtColor(optimized_img, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb_img)
            
            # BLIP2 분석
            inputs = self.blip_processor(pil_img, prompt, return_tensors="pt").to(self.device)
            
            # FP16 처리
            if self.device.type == 'cuda':
                for key in inputs:
                    if inputs[key].dtype == torch.float32:
                        inputs[key] = inputs[key].half()
            
            with torch.no_grad():
                outputs = self.blip_model.generate(**inputs, max_length=50, do_sample=False)
            
            result = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
            return result
            
        except Exception as e:
            self.get_logger().error(f"BLIP2 분석 오류: {str(e)}")
            return "unknown"
    
    def get_default_properties(self):
        """기본 물리적 특성"""
        return {
            'weight': 'medium',
            'temperature': 'room_temp',
            'stability': 'stable',
            'fragility': 'medium',
            'size': 'medium',
            'movement_speed': 'static'
        }

    def scan_callback(self, msg: LaserScan):
        """LiDAR 콜백: 최신 스캔 저장"""
        self.latest_scan = msg

    def analyze_object_by_computer_vision(self, image, class_name, bbox):
        """컴퓨터 비전 기반으로 물체의 물리적 특성 분석"""
        try:
            # 바운딩 박스 영역 추출
            x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
            object_image = image[y1:y2, x1:x2]
            
            if object_image.size == 0:
                return self.get_default_properties()
            
            properties = {}
            
            # 1. 크기 분석 (이미지 영역 기반)
            properties['size'] = self.analyze_size_by_area(object_image)
            
            # 2. 색상 기반 온도 추정
            properties['temperature'] = self.analyze_temperature_by_color(object_image)
            
            # 3. 텍스처 기반 재질 분석
            properties['fragility'] = self.analyze_fragility_by_texture(object_image)
            
            # 4. 형태 기반 안정성 분석
            properties['stability'] = self.analyze_stability_by_shape(object_image, class_name)
            
            # 5. 무게 추정 (크기와 재질 기반)
            properties['weight'] = self.estimate_weight_by_size_and_material(properties['size'], properties['fragility'])
            
            # 6. 움직임 속도 추정
            properties['movement_speed'] = self.estimate_movement_speed(class_name, properties['stability'])
            
            return properties
            
        except Exception as e:
            self.get_logger().error(f"컴퓨터 비전 분석 오류: {str(e)}")
            return self.get_default_properties()
    
    def analyze_size_by_area(self, image):
        """이미지 영역 기반 크기 분석"""
        height, width = image.shape[:2]
        area = height * width
        
        if area < 1000:
            return 'very_small'
        elif area < 5000:
            return 'small'
        elif area < 20000:
            return 'medium'
        elif area < 50000:
            return 'large'
        else:
            return 'very_large'
    
    def analyze_temperature_by_color(self, image):
        """색상 기반 온도 추정"""
        try:
            # HSV 색상 공간으로 변환
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 빨간색/주황색 영역 탐지 (뜨거운 물체)
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = red_mask1 + red_mask2
            
            # 노란색 영역 탐지 (따뜻한 물체)
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # 파란색 영역 탐지 (차가운 물체)
            lower_blue = np.array([100, 100, 100])
            upper_blue = np.array([130, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # 각 색상 영역의 비율 계산
            total_pixels = image.shape[0] * image.shape[1]
            red_ratio = np.sum(red_mask > 0) / total_pixels
            yellow_ratio = np.sum(yellow_mask > 0) / total_pixels
            blue_ratio = np.sum(blue_mask > 0) / total_pixels
            
            # 온도 추정
            if red_ratio > 0.3:
                return 'very_hot'
            elif red_ratio > 0.1 or yellow_ratio > 0.2:
                return 'hot'
            elif yellow_ratio > 0.1:
                return 'warm'
            elif blue_ratio > 0.2:
                return 'cold'
            else:
                return 'room_temp'
                
        except Exception as e:
            return 'room_temp'
    
    def analyze_fragility_by_texture(self, image):
        """텍스처 기반 재질 분석"""
        try:
            # 그레이스케일 변환
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 엣지 검출 (깨지기 쉬운 물체는 더 많은 엣지를 가짐)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
            
            # 텍스처 분석 (GLCM 특성)
            # 간단한 방법: 표준편차로 텍스처 복잡도 측정
            texture_std = np.std(gray)
            
            # 재질 추정
            if edge_density > 0.1 or texture_std > 50:
                return 'very_high'  # 복잡한 텍스처 = 깨지기 쉬움
            elif edge_density > 0.05 or texture_std > 30:
                return 'high'
            elif edge_density > 0.02 or texture_std > 15:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            return 'medium'
    
    def analyze_stability_by_shape(self, image, class_name):
        """형태 기반 안정성 분석"""
        try:
            # 그레이스케일 변환
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 이진화
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # 윤곽선 찾기
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 'stable'
            
            # 가장 큰 윤곽선 선택
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 윤곽선의 특성 분석
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # 원형도 계산 (1에 가까울수록 원형)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # 직사각형 피팅
            rect = cv2.minAreaRect(largest_contour)
            width, height = rect[1]
            aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 1
            
            # 안정성 추정
            if circularity > 0.8:
                return 'stable'  # 원형 = 안정적
            elif aspect_ratio > 3:
                return 'unstable'  # 매우 길쭉함 = 불안정
            elif aspect_ratio > 2:
                return 'movable'  # 길쭉함 = 움직일 수 있음
            else:
                return 'stable'
                
        except Exception as e:
            return 'stable'
    
    def estimate_weight_by_size_and_material(self, size, fragility):
        """크기와 재질 기반 무게 추정"""
        # 크기별 기본 무게
        size_weights = {
            'very_small': 0.1,
            'small': 0.5,
            'medium': 2.0,
            'large': 8.0,
            'very_large': 25.0
        }
        
        # 재질별 밀도 가중치
        fragility_weights = {
            'very_high': 0.8,  # 유리 등 - 가벼움
            'high': 1.0,        # 플라스틱 등 - 보통
            'medium': 1.2,      # 나무 등 - 약간 무거움
            'low': 1.5          # 금속 등 - 무거움
        }
        
        base_weight = size_weights.get(size, 2.0)
        density_factor = fragility_weights.get(fragility, 1.0)
        
        estimated_weight = base_weight * density_factor
        
        # 무게 등급 분류
        if estimated_weight < 0.5:
            return 'very_light'
        elif estimated_weight < 2.0:
            return 'light'
        elif estimated_weight < 8.0:
            return 'medium'
        elif estimated_weight < 20.0:
            return 'heavy'
        else:
            return 'very_heavy'
    
    def estimate_movement_speed(self, class_name, stability):
        """클래스명과 안정성 기반 움직임 속도 추정"""
        if class_name in ['person', 'human']:
            return 'variable'
        elif stability == 'dynamic':
            return 'fast'
        elif stability == 'movable':
            return 'slow'
        elif stability == 'unstable':
            return 'variable'
        else:
            return 'static'

    def analyze_weight_and_risk_properties(self, image, class_name, bbox):
        """BLIP2를 사용하여 물체의 무게와 위험도를 한 번에 분석"""
        try:
            # 바운딩 박스 영역 추출
            x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
            object_image = image[y1:y2, x1:x2]
            
            if object_image.size == 0:
                return self.get_default_weight_risk_properties()
            
            # BLIP2로 무게와 위험도를 한 번에 분석
            properties = self.analyze_weight_and_risk_combined(object_image, class_name)
            
            return properties
            
        except Exception as e:
            self.get_logger().error(f"무게 및 위험도 특성 분석 오류: {str(e)}")
            return self.get_default_weight_risk_properties()
    
    def get_default_weight_risk_properties(self):
        """기본 무게 및 위험도 특성"""
        return {
            'weight': 'medium',
            'risk_type': 'static',
            'mobility': 'stationary',
            'stability': 'stable',
            'danger_level': 'medium_risk',
            'size': 'medium'
        }

    def analyze_weight_and_risk_combined(self, image, class_name):
        """BLIP2로 무게와 위험도를 한 번에 분석"""
        try:
            prompt = f"""Analyze this {class_name} object comprehensively. Provide the following information in a structured format:

1. Weight category: very_light (under 0.5kg), light (0.5-2kg), medium (2-8kg), heavy (8-20kg), or very_heavy (over 20kg)
2. Risk type: static (stationary), dynamic (moving), human (person), sudden (unexpected), or environmental (part of environment)
3. Mobility: stationary (fixed), movable (can be moved), mobile (can move), or unpredictable (unexpected movement)
4. Danger level: safe, low_risk, medium_risk, high_risk, or extremely_dangerous
5. Size: very_small, small, medium, large, or very_large

Consider the object's material, size, potential for movement, and risk to navigation. Format your response as: weight|risk_type|mobility|danger_level|size"""

            result = self.blip_analysis(image, prompt)
            
            # 결과 파싱
            properties = self.parse_combined_analysis(result)
            
            self.get_logger().debug(f"통합 분석 결과 - {class_name}: {properties}")
            
            return properties
            
        except Exception as e:
            self.get_logger().error(f"통합 분석 실패: {str(e)}")
            return self.get_default_weight_risk_properties()
    
    def parse_combined_analysis(self, result):
        """통합 분석 결과를 파싱"""
        try:
            # 결과를 파이프(|)로 분리
            parts = result.strip().split('|')
            
            if len(parts) >= 5:
                weight = self.parse_weight(parts[0].strip())
                risk_type = self.parse_risk_type(parts[1].strip())
                mobility = self.parse_mobility(parts[2].strip())
                danger_level = self.parse_danger_level(parts[3].strip())
                size = self.parse_size(parts[4].strip())
            else:
                # 파싱 실패 시 기본값 사용
                weight = 'medium'
                risk_type = 'static'
                mobility = 'stationary'
                danger_level = 'medium_risk'
                size = 'medium'
            
            return {
                'weight': weight,
                'risk_type': risk_type,
                'mobility': mobility,
                'stability': self.infer_stability_from_mobility(mobility),
                'danger_level': danger_level,
                'size': size
            }
            
        except Exception as e:
            self.get_logger().error(f"결과 파싱 실패: {str(e)}")
            return self.get_default_weight_risk_properties()
    
    def parse_weight(self, text):
        """무게 텍스트 파싱"""
        text_lower = text.lower()
        if 'very_light' in text_lower or 'under 0.5' in text_lower:
            return 'very_light'
        elif 'light' in text_lower or '0.5-2' in text_lower:
            return 'light'
        elif 'medium' in text_lower or '2-8' in text_lower:
            return 'medium'
        elif 'heavy' in text_lower or '8-20' in text_lower:
            return 'heavy'
        elif 'very_heavy' in text_lower or 'over 20' in text_lower:
            return 'very_heavy'
        else:
            return 'medium'
    
    def parse_risk_type(self, text):
        """위험 유형 텍스트 파싱"""
        text_lower = text.lower()
        if 'human' in text_lower or 'person' in text_lower:
            return 'human'
        elif 'dynamic' in text_lower or 'moving' in text_lower:
            return 'dynamic'
        elif 'sudden' in text_lower or 'unexpected' in text_lower:
            return 'sudden'
        elif 'environmental' in text_lower or 'environment' in text_lower:
            return 'environmental'
        else:
            return 'static'
    
    def parse_mobility(self, text):
        """이동성 텍스트 파싱"""
        text_lower = text.lower()
        if 'stationary' in text_lower or 'fixed' in text_lower:
            return 'stationary'
        elif 'movable' in text_lower or 'can be moved' in text_lower:
            return 'movable'
        elif 'mobile' in text_lower or 'can move' in text_lower:
            return 'mobile'
        elif 'unpredictable' in text_lower or 'unexpected' in text_lower:
            return 'unpredictable'
        else:
            return 'stationary'
    
    def parse_danger_level(self, text):
        """위험도 레벨 텍스트 파싱"""
        text_lower = text.lower()
        if 'extremely_dangerous' in text_lower or 'very dangerous' in text_lower:
            return 'extremely_dangerous'
        elif 'high_risk' in text_lower or 'dangerous' in text_lower:
            return 'high_risk'
        elif 'medium_risk' in text_lower or 'moderate' in text_lower:
            return 'medium_risk'
        elif 'low_risk' in text_lower or 'minimal' in text_lower:
            return 'low_risk'
        elif 'safe' in text_lower or 'harmless' in text_lower:
            return 'safe'
        else:
            return 'medium_risk'
    
    def parse_size(self, text):
        """크기 텍스트 파싱"""
        text_lower = text.lower()
        if 'very_small' in text_lower or 'tiny' in text_lower:
            return 'very_small'
        elif 'small' in text_lower:
            return 'small'
        elif 'medium' in text_lower:
            return 'medium'
        elif 'large' in text_lower:
            return 'large'
        elif 'very_large' in text_lower or 'huge' in text_lower:
            return 'very_large'
        else:
            return 'medium'
    
    def infer_stability_from_mobility(self, mobility):
        """이동성으로부터 안정성 추론"""
        if mobility == 'stationary':
            return 'stable'
        elif mobility == 'movable':
            return 'stable'
        elif mobility == 'mobile':
            return 'dynamic'
        elif mobility == 'unpredictable':
            return 'unstable'
        else:
            return 'stable'

def main(args=None):
    rclpy.init(args=args)
    node = None
    
    try:
        node = YOLOBlip2RiskNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"노드 실행 오류: {str(e)}")
    finally:
        # 비동기 처리 정리
        if node is not None and hasattr(node, 'thread_executor'):
            node.thread_executor.shutdown(wait=True)
        # rclpy 종료 전에 컨텍스트 확인
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception as e:
            print(f"rclpy 종료 오류 (무시됨): {str(e)}")

if __name__ == '__main__':
    main()
 