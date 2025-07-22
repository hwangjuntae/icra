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
        self.model_dir = Path("/root/ws/src/risk_nav/models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.yolo_model_path = self.model_dir / "yolo11n.pt"
        
        # 모델 초기화
        self.initialize_models()
        
        # 위험도 카테고리 설정
        self.risk_categories = {
            "person": 40,
            "chair": 20,
            "desk": 15,
            "laptop": 25,
            "monitor": 20,
            "keyboard": 10,
            "mouse": 5,
            "bottle": 30,
            "cup": 35,
            "scissors": 70,
            "knife": 90,
            "fire extinguisher": 10,
            "default": 25
        }
        
        # 객체 인식 최적화 설정
        self.confidence_threshold = 0.3  # 신뢰도 임계값 낮춤 (더 많은 객체 인식)
        self.nms_threshold = 0.4  # NMS 임계값
        self.max_detections = 20  # 최대 탐지 객체 수
        
        # 거리 기반 필터링 설정
        self.max_distance = 5.0  # 최대 거리 (미터)
        self.min_distance = 0.3  # 최소 거리 (미터)
        self.distance_weight = 0.7  # 거리 가중치
        
        self.get_logger().info("YOLO + BLIP2 위험도 평가 노드 시작 (최적화 버전)")
        self.get_logger().info("구독 토픽: /Camera/rgb, /Lidar/laser_scan")
        self.get_logger().info("발행 토픽: /risk_assessment/image (sensor_msgs/Image)")
        self.get_logger().info(f"목표 처리 주파수: {self.target_fps}Hz ({self.frame_interval*1000:.1f}ms 간격)")
        self.get_logger().info(f"처리 해상도 스케일: {self.processing_scale:.1f} (최소: {self.min_processing_width}x{self.min_processing_height})")
        self.get_logger().info(f"YOLO 설정: conf={self.confidence_threshold}, nms={self.nms_threshold}, max_det={self.max_detections}")
        self.get_logger().info(f"거리 필터링: {self.min_distance}-{self.max_distance}m, 가중치={self.distance_weight}")
        self.get_logger().info(f"BLIP2 최적화: 배치크기={self.blip_batch_size}, 해상도={self.blip_resolution}, 캐시크기={self.cache_max_size}")
        self.get_logger().info(f"모델 저장 경로: {self.model_dir}")
        
    def initialize_models(self):
        """모델 초기화"""
        try:
            if YOLO_AVAILABLE:
                # YOLOv11 모델 로드
                self.get_logger().info(f"YOLOv11 모델 로드 중... 경로: {self.yolo_model_path}")
                
                # 모델이 존재하지 않으면 다운로드
                if not self.yolo_model_path.exists():
                    self.get_logger().info("YOLOv11 모델이 없습니다. 다운로드 중...")
                    
                # YOLOv11 nano 모델 로드 (자동 다운로드)
                self.yolo_model = YOLO('yolo11n.pt')
                
                # 모델을 지정된 경로로 이동
                if not self.yolo_model_path.exists():
                    import shutil
                    # ultralytics가 다운로드한 모델을 우리 경로로 복사
                    home_model_path = Path.home() / '.ultralytics' / 'models' / 'yolo11n.pt'
                    if home_model_path.exists():
                        shutil.copy2(home_model_path, self.yolo_model_path)
                        self.get_logger().info(f"모델을 {self.yolo_model_path}로 복사 완료")
                
                # 지정된 경로에서 모델 로드
                if self.yolo_model_path.exists():
                    self.yolo_model = YOLO(str(self.yolo_model_path))
                    self.get_logger().info(f"YOLOv11 모델 로드 완료: {self.yolo_model_path}")
                else:
                    self.get_logger().info("YOLOv11 모델 로드 완료 (기본 위치)")
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
                    
                    self.get_logger().info(f"위험도 평가: {len(risk_results.get('detections', []))}개 객체 | "
                                         f"처리: {processing_time:.1f}ms | FPS: {actual_fps:.1f} | "
                                         f"해상도: {processed_image.shape[1]}x{processed_image.shape[0]} | "
                                         f"BLIP 대기: {blip_queue_size} | {cache_stats}")
                    
                except Exception as bridge_error:
                    self.get_logger().error(f"이미지 변환 실패: {str(bridge_error)}")
            else:
                self.get_logger().error("cv_bridge를 사용할 수 없어 이미지 전송 실패")
            
        except Exception as e:
            self.get_logger().error(f"이미지 처리 오류: {str(e)}")
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
            
            label = f"{class_name} {confidence:.2f}"
            risk_text = f"Risk: {risk_score:.1f} ({risk_level})"
            distance_text = f"Dist: {distance}m" if isinstance(distance, (int, float)) else f"Dist: {distance}"
            
            # 텍스트 배경 박스 크기 계산
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            (risk_w, risk_h), _ = cv2.getTextSize(risk_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            (dist_w, dist_h), _ = cv2.getTextSize(distance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            
            # 텍스트 배경 박스 그리기
            max_text_w = max(label_w, risk_w, dist_w)
            cv2.rectangle(vis_image, (x1, y1 - label_h - risk_h - dist_h - 15), 
                         (x1 + max_text_w + 10, y1), color, -1)
            
            # 텍스트 그리기
            cv2.putText(vis_image, label, (x1 + 5, y1 - dist_h - risk_h - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(vis_image, risk_text, (x1 + 5, y1 - dist_h - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            cv2.putText(vis_image, distance_text, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # 전체 위험도 정보 표시
        overall_risk = risk_results.get('overall_risk_score', 0)
        overall_level = risk_results.get('risk_level', 'low')
        detection_count = len(risk_results.get('detections', []))
        
        # 상단에 전체 정보 표시
        info_text = f"Objects: {detection_count} | Overall Risk: {overall_risk:.1f} ({overall_level})"
        
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
            cv2.putText(vis_image, scene_desc, (15, height - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_image
            
    def assess_risk(self, image, scale_info=None):
        """위험도 평가 수행 (LiDAR 융합 포함)"""
        results = {
            "timestamp": time.time(),
            "detections": [],
            "overall_risk_score": 0.0,
            "risk_level": "low",
            "scene_description": "장면 분석 중..."
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
                        
                        # 위험도 점수 계산 (거리 가중치 적용)
                        risk_score = self.calculate_risk_score_with_distance(class_name, confidence, distance)
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
                    
                    # 거리 기반 정렬 (가까운 객체 우선)
                    detections.sort(key=lambda d: d.get('distance', float('inf')) if d.get('distance') is not None else float('inf'))
                    
                    # 최대 탐지 수 제한
                    detections = detections[:self.max_detections]
                    
                    results["detections"] = detections
                    
                    if detection_count > 0:
                        results["overall_risk_score"] = total_risk / detection_count
                        results["risk_level"] = self.get_risk_level(results["overall_risk_score"])
                        
                        self.get_logger().info(f"YOLO 탐지: {detection_count}개 객체 (거리순 정렬)")
            
            # 배치 처리로 BLIP2 분석 수행
            if self.blip_model is not None and self.blip_queue:
                # 비동기로 배치 처리 시작
                self.thread_executor.submit(self.process_blip_batch, self.blip_queue.copy(), results)
                self.blip_queue.clear()
                
                # 전체 장면 설명
                results["scene_description"] = self.get_scene_description(image)
                
        except Exception as e:
            self.get_logger().error(f"위험도 평가 오류: {str(e)}")
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
        
    def calculate_risk_score_with_distance(self, class_name, confidence, distance):
        """거리 가중치를 적용한 위험도 점수 계산"""
        # 기본 위험도 점수
        base_score = self.calculate_risk_score(class_name, confidence)
        
        # 거리 가중치 적용
        if distance is not None:
            # 거리가 가까울수록 위험도 증가
            distance_factor = max(0.5, 2.0 - (distance / self.max_distance))
            base_score *= distance_factor
            
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
                # 위험도 평가 특화 프롬프트
                prompt = f"Is this {obj_info['class_name']} object dangerous or safe? Explain the safety risk level."
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
            
    def calculate_blip_risk_score(self, description):
        """BLIP2 설명에서 위험도 점수 계산"""
        description_lower = description.lower()
        
        # 위험 키워드 점수
        danger_keywords = {
            'dangerous': 80, 'unsafe': 70, 'hazardous': 85, 'risky': 60,
            'sharp': 75, 'hot': 70, 'fire': 90, 'toxic': 95, 'poisonous': 90,
            'knife': 85, 'blade': 80, 'cutting': 70, 'weapon': 90
        }
        
        safe_keywords = {
            'safe': -30, 'harmless': -25, 'secure': -20, 'non-toxic': -40,
            'blunt': -20, 'soft': -15, 'gentle': -10
        }
        
        base_score = 25  # 기본 점수
        
        for keyword, score in danger_keywords.items():
            if keyword in description_lower:
                base_score += score * 0.5  # 가중치 적용
                
        for keyword, score in safe_keywords.items():
            if keyword in description_lower:
                base_score += score * 0.3  # 가중치 적용
                
        return max(0, min(100, base_score))
        
    def calculate_risk_score(self, class_name, confidence):
        """위험도 점수 계산"""
        base_score = self.risk_categories.get(class_name, self.risk_categories["default"])
        
        # 신뢰도 가중치 적용
        confidence_weight = confidence * 20
        
        # 특정 객체별 위험도 조정
        if class_name in ["scissors", "knife"]:
            adjustment = 30
        elif class_name == "fire extinguisher":
            adjustment = -20
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
        """전체 장면 설명 생성 (최적화됨)"""
        if self.blip_model is None:
            return "장면 설명을 생성할 수 없습니다."
            
        try:
            # 해상도 최적화
            optimized_img = self.optimize_image_for_blip(image)
            rgb_image = cv2.cvtColor(optimized_img, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb_image)
            
            # 안전성 특화 프롬프트
            prompt = "Describe the overall safety situation in this scene. Are there any potential hazards?"
            
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

    def scan_callback(self, msg: LaserScan):
        """LiDAR 콜백: 최신 스캔 저장"""
        self.latest_scan = msg

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
 