#!/usr/bin/env python3
"""
위험도 맵 생성 ROS2 노드 (오도메트리 기반 + YOLO+BLIP2 위험도 감지)
감지된 위험을 기반으로 2D 위험도 맵을 작성하고 시각화
로봇의 이동에 따라 위험도 맵이 업데이트됨
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped, PoseWithCovariance
from std_msgs.msg import Header
import cv2
import numpy as np
import json
import time
import traceback
import os
from pathlib import Path
import math
from collections import deque
import tf2_ros
from tf2_ros import TransformBroadcaster
import tf2_geometry_msgs
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

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

class RiskMapNode(Node):
    def __init__(self):
        super().__init__('risk_map_node')
        
        # cv_bridge 초기화
        if CV_BRIDGE_AVAILABLE:
            self.bridge = CvBridge()
        else:
            self.bridge = None
            self.get_logger().error("cv_bridge를 사용할 수 없습니다.")
        
        # 카메라 이미지 구독 (직접 카메라 데이터 처리)
        self.camera_subscription = self.create_subscription(
            Image,
            '/Camera/rgb',
            self.camera_callback,
            10
        )
        
        # LiDAR 구독
        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/Lidar/laser_scan',
            self.scan_callback,
            10
        )
        
        # 오도메트리 구독
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        # 위험도 맵 발행
        self.risk_map_publisher = self.create_publisher(
            OccupancyGrid,
            '/risk_map',
            10
        )
        
        # 위험도 맵 시각화 이미지 발행
        self.risk_map_viz_publisher = self.create_publisher(
            Image,
            '/risk_map/visualization',
            10
        )
        
        # TF 브로드캐스터
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # 맵 설정
        self.map_resolution = 0.1  # 미터/픽셀
        self.map_width = 400  # 픽셀 (40m) - 더 큰 맵
        self.map_height = 400  # 픽셀 (40m) - 더 큰 맵
        self.map_center_x = 200  # 픽셀
        self.map_center_y = 200  # 픽셀
        
        # 위험도 맵 초기화
        self.risk_map = np.zeros((self.map_height, self.map_width), dtype=np.float32)
        self.risk_decay_rate = 0.98  # 위험도 감쇠율 (더 천천히)
        self.risk_threshold = 0.05  # 위험도 임계값
        
        # 로봇 위치 추적
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.last_robot_x = 0.0
        self.last_robot_y = 0.0
        self.last_robot_theta = 0.0
        
        # 위험 객체 히스토리 (전역 좌표)
        self.risk_history = deque(maxlen=200)
        
        # 맵 변환 히스토리
        self.map_transforms = deque(maxlen=50)
        
        # 위험도 레벨별 색상 매핑
        self.risk_colors = {
            "low": (0, 255, 0),      # 녹색
            "medium": (0, 255, 255), # 노란색
            "high": (0, 0, 255)      # 빨간색
        }
        
        # YOLO+BLIP2 관련 설정
        self.latest_image = None
        self.latest_scan = None
        self.processing_scale = 0.8
        self.min_processing_width = 640
        self.min_processing_height = 480
        
        # YOLO 설정
        self.confidence_threshold = 0.3
        self.nms_threshold = 0.4
        self.max_detections = 20
        
        # 거리 기반 필터링 설정
        self.max_distance = 5.0
        self.min_distance = 0.3
        self.distance_weight = 0.7
        self.camera_hfov = 90.0
        
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
        
        # BLIP2 최적화 설정
        self.blip_batch_size = 4
        self.blip_resolution = (224, 224)
        self.thread_executor = ThreadPoolExecutor(max_workers=2)
        
        # 스마트 캐싱
        self.risk_cache = OrderedDict()
        self.cache_max_size = 100
        self.cache_hit_count = 0
        self.cache_total_count = 0
        
        # 모델 초기화
        self.initialize_models()
        
        # 타이머 생성 (맵 업데이트)
        self.map_update_timer = self.create_timer(
            0.5,  # 2Hz로 맵 업데이트 (더 빠른 업데이트)
            self.update_risk_map
        )
        
        # 맵 프레임 설정
        self.map_frame = "map"
        self.base_frame = "base_link"
        self.odom_frame = "odom"
        
        self.get_logger().info("위험도 맵 생성 노드 시작 (오도메트리 기반 + YOLO+BLIP2)")
        self.get_logger().info("구독 토픽: /Camera/rgb, /Lidar/laser_scan, /odom")
        self.get_logger().info("발행 토픽: /risk_map, /risk_map/visualization")
        self.get_logger().info(f"맵 해상도: {self.map_resolution}m/픽셀")
        self.get_logger().info(f"맵 크기: {self.map_width}x{self.map_height} 픽셀 ({self.map_width*self.map_resolution:.1f}m x {self.map_height*self.map_resolution:.1f}m)")
        self.get_logger().info(f"YOLO 설정: conf={self.confidence_threshold}, nms={self.nms_threshold}, max_det={self.max_detections}")
        self.get_logger().info(f"거리 필터링: {self.min_distance}-{self.max_distance}m, 가중치={self.distance_weight}")
        
    def initialize_models(self):
        """YOLO+BLIP2 모델 초기화"""
        try:
            if YOLO_AVAILABLE:
                # YOLOv11 모델 로드
                self.get_logger().info("YOLOv11 모델 로드 중...")
                model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'yolo11n.pt')
                self.yolo_model = YOLO(model_path)
                self.get_logger().info("YOLOv11 모델 로드 완료")
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
                
                self.get_logger().info(f"BLIP2 모델 로드 완료 (디바이스: {self.device})")
            else:
                self.blip_processor = None
                self.blip_model = None
                self.device = torch.device('cpu')
                
        except Exception as e:
            self.get_logger().error(f"모델 초기화 실패: {str(e)}")
            import traceback
            traceback.print_exc()
        
    def camera_callback(self, msg):
        """카메라 이미지 콜백"""
        try:
            if self.bridge is not None:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                self.latest_image = cv_image
                
                # 위험도 평가 수행
                if self.latest_image is not None:
                    self.process_image_for_risk_assessment()
                    
        except Exception as e:
            self.get_logger().error(f"카메라 이미지 처리 오류: {str(e)}")
            
    def process_image_for_risk_assessment(self):
        """이미지에서 위험도 평가 수행"""
        try:
            if self.latest_image is None or self.yolo_model is None:
                return
                
            # 이미지 다운샘플링
            processed_image, scale_info = self.downsample_image(self.latest_image)
            
            # YOLO 객체 탐지
            detections = self.perform_yolo_detection(processed_image, scale_info)
            
            # 위험 객체를 맵에 추가
            if detections:
                self.add_detections_to_map(detections)
                
        except Exception as e:
            self.get_logger().error(f"위험도 평가 처리 오류: {str(e)}")
            traceback.print_exc()
            
    def downsample_image(self, image):
        """이미지 다운샘플링"""
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
        
    def perform_yolo_detection(self, image, scale_info):
        """YOLO 객체 탐지 수행"""
        detections = []
        
        try:
            # YOLO 추론 수행
            yolo_results = self.yolo_model(
                image,
                conf=self.confidence_threshold,
                iou=self.nms_threshold,
                max_det=self.max_detections,
                verbose=False
            )
            
            if len(yolo_results) > 0 and yolo_results[0].boxes is not None:
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
                    
                    # 위험도 점수 계산
                    risk_score = self.calculate_risk_score_with_distance(class_name, confidence, distance)
                    risk_level = self.get_risk_level(risk_score)
                    
                    detection_result = {
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                        "risk_score": float(risk_score),
                        "risk_level": risk_level,
                        "distance": distance,
                        "timestamp": time.time()
                    }
                    
                    detections.append(detection_result)
                    
                # 거리 기반 정렬 (가까운 객체 우선)
                detections.sort(key=lambda d: d.get('distance', float('inf')) if d.get('distance') is not None else float('inf'))
                
                # 최대 탐지 수 제한
                detections = detections[:self.max_detections]
                
                self.get_logger().info(f"YOLO 탐지: {len(detections)}개 객체")
                
        except Exception as e:
            self.get_logger().error(f"YOLO 탐지 오류: {str(e)}")
            
        return detections
        
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
        
    def add_detections_to_map(self, detections):
        """탐지된 객체들을 맵에 추가"""
        try:
            for detection in detections:
                # 이미지 좌표를 로봇 기준 상대 좌표로 변환
                bbox = detection['bbox']
                center_x = (bbox['x1'] + bbox['x2']) / 2.0
                center_y = (bbox['y1'] + bbox['y2']) / 2.0
                
                robot_x, robot_y = self.image_to_robot_coordinates(center_x, center_y)
                
                # 로봇 기준 좌표를 전역 좌표로 변환
                global_x = self.robot_x + robot_x
                global_y = self.robot_y + robot_y
                
                # 전역 좌표를 맵 좌표로 변환
                map_x, map_y = self.global_to_map_coordinates(global_x, global_y)
                
                if 0 <= map_x < self.map_width and 0 <= map_y < self.map_height:
                    # 위험도 점수를 맵에 추가
                    risk_value = detection['risk_score'] / 100.0  # 0-1 범위로 정규화
                    
                    # 가우시안 블러로 부드러운 위험도 분포 생성
                    width = bbox['x2'] - bbox['x1']
                    height = bbox['y2'] - bbox['y1']
                    radius = max(3, int(max(width, height) / 20))  # 객체 크기에 비례한 반지름
                    
                    # 위험도 분포 생성
                    y, x = np.ogrid[:self.map_height, :self.map_width]
                    mask = (x - map_x)**2 + (y - map_y)**2 <= radius**2
                    
                    # 거리에 따른 위험도 감쇠
                    distance = np.sqrt((x - map_x)**2 + (y - map_y)**2)
                    decay = np.exp(-distance / (radius * 0.5))
                    decay = np.where(mask, decay, 0)
                    
                    # 맵에 위험도 추가
                    self.risk_map += risk_value * decay
                    
                    # 히스토리에 추가 (전역 좌표)
                    self.risk_history.append({
                        'global_x': global_x,
                        'global_y': global_y,
                        'map_x': map_x,
                        'map_y': map_y,
                        'risk_value': risk_value,
                        'risk_level': detection['risk_level'],
                        'class_name': detection['class_name'],
                        'timestamp': detection['timestamp']
                    })
                    
        except Exception as e:
            self.get_logger().error(f"탐지 객체 맵 추가 오류: {str(e)}")
        
    def odom_callback(self, msg: Odometry):
        """오도메트리 콜백"""
        try:
            # 로봇의 현재 위치 업데이트
            self.robot_x = msg.pose.pose.position.x
            self.robot_y = msg.pose.pose.position.y
            
            # 쿼터니언을 오일러 각도로 변환
            q = msg.pose.pose.orientation
            self.robot_theta = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                        1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            
            # 로봇이 이동했는지 확인
            dx = self.robot_x - self.last_robot_x
            dy = self.robot_y - self.last_robot_y
            dtheta = self.robot_theta - self.last_robot_theta
            
            # 최소 이동 거리 임계값
            min_movement = 0.05  # 5cm
            min_rotation = 0.1   # 약 6도
            
            if abs(dx) > min_movement or abs(dy) > min_movement or abs(dtheta) > min_rotation:
                # 맵 변환 수행
                self.transform_risk_map(dx, dy, dtheta)
                
                # 이전 위치 업데이트
                self.last_robot_x = self.robot_x
                self.last_robot_y = self.robot_y
                self.last_robot_theta = self.robot_theta
                
                # 변환 히스토리 저장
                self.map_transforms.append({
                    'dx': dx,
                    'dy': dy,
                    'dtheta': dtheta,
                    'timestamp': time.time()
                })
                
        except Exception as e:
            self.get_logger().error(f"오도메트리 처리 오류: {str(e)}")
            
    def transform_risk_map(self, dx, dy, dtheta):
        """로봇 이동에 따른 위험도 맵 변환"""
        try:
            # 맵 좌표계에서의 변환량 계산
            map_dx = int(dx / self.map_resolution)
            map_dy = int(dy / self.map_resolution)
            
            # 회전 변환 행렬 생성
            cos_theta = math.cos(-dtheta)  # 반시계 방향
            sin_theta = math.sin(-dtheta)
            
            # 맵 중심을 기준으로 회전 변환
            center_x, center_y = self.map_center_x, self.map_center_y
            
            # 새로운 맵 생성
            new_risk_map = np.zeros_like(self.risk_map)
            
            for y in range(self.map_height):
                for x in range(self.map_width):
                    # 맵 중심을 기준으로 상대 좌표
                    rel_x = x - center_x
                    rel_y = y - center_y
                    
                    # 회전 변환
                    new_rel_x = rel_x * cos_theta - rel_y * sin_theta
                    new_rel_y = rel_x * sin_theta + rel_y * cos_theta
                    
                    # 평행 이동
                    new_x = int(new_rel_x + center_x - map_dx)
                    new_y = int(new_rel_y + center_y - map_dy)
                    
                    # 범위 확인
                    if 0 <= new_x < self.map_width and 0 <= new_y < self.map_height:
                        new_risk_map[y, x] = self.risk_map[new_y, new_x]
                        
            # 변환된 맵으로 업데이트
            self.risk_map = new_risk_map
            
            # 위험 히스토리도 변환
            self.transform_risk_history(dx, dy, dtheta)
            
        except Exception as e:
            self.get_logger().error(f"맵 변환 오류: {str(e)}")
            
    def transform_risk_history(self, dx, dy, dtheta):
        """위험 히스토리 변환"""
        try:
            for risk_obj in self.risk_history:
                # 전역 좌표에서 변환
                x, y = risk_obj['global_x'], risk_obj['global_y']
                
                # 회전 변환
                cos_theta = math.cos(-dtheta)
                sin_theta = math.sin(-dtheta)
                
                new_x = x * cos_theta - y * sin_theta
                new_y = x * sin_theta + y * cos_theta
                
                # 평행 이동
                new_x -= dx
                new_y -= dy
                
                # 업데이트
                risk_obj['global_x'] = new_x
                risk_obj['global_y'] = new_y
                
        except Exception as e:
            self.get_logger().error(f"히스토리 변환 오류: {str(e)}")
            
    def image_to_robot_coordinates(self, img_x, img_y):
        """이미지 좌표를 로봇 기준 좌표로 변환"""
        # 이미지 중심을 로봇 기준으로 변환
        # 이미지: (0,0) -> (1280,720)
        # 로봇: 전방이 양의 X축, 좌측이 양의 Y축
        
        # 정규화된 좌표 (-1 to 1)
        norm_x = (img_x - 640) / 640.0
        norm_y = (img_y - 360) / 360.0
        
        # 카메라 시야각을 고려한 거리 추정 (간단한 모델)
        # 화면 중앙에서 멀수록 실제 거리가 멀다고 가정
        distance_factor = 1.0 + abs(norm_x) + abs(norm_y)
        
        # 로봇 기준 좌표 (미터 단위)
        robot_x = distance_factor * 2.0  # 기본 2m 거리
        robot_y = norm_x * 3.0  # 좌우 3m 범위
        
        return robot_x, robot_y
        
    def global_to_map_coordinates(self, global_x, global_y):
        """전역 좌표를 맵 좌표로 변환"""
        # 맵 중심을 (0,0)으로 하는 좌표계
        map_x = int(self.map_center_x + global_x / self.map_resolution)
        map_y = int(self.map_center_y - global_y / self.map_resolution)  # Y축 반전
        
        return map_x, map_y
        
    def update_risk_map(self):
        """위험도 맵 업데이트 및 발행"""
        try:
            # 위험도 감쇠 적용
            self.risk_map *= self.risk_decay_rate
            
            # 위험도 맵을 OccupancyGrid로 변환
            occupancy_grid = self.create_occupancy_grid()
            self.risk_map_publisher.publish(occupancy_grid)
            
            # 시각화 이미지 생성 및 발행
            viz_image = self.create_risk_map_visualization()
            if self.bridge is not None:
                viz_msg = self.bridge.cv2_to_imgmsg(viz_image, "bgr8")
                self.risk_map_viz_publisher.publish(viz_msg)
                
            # TF 브로드캐스트
            self.broadcast_map_transform()
            
            # 통계 정보 로그
            max_risk = np.max(self.risk_map)
            avg_risk = np.mean(self.risk_map)
            risk_pixels = np.sum(self.risk_map > self.risk_threshold)
            
            self.get_logger().info(f"위험도 맵 업데이트: 최대={max_risk:.3f}, 평균={avg_risk:.3f}, "
                                 f"위험픽셀={risk_pixels}, 히스토리={len(self.risk_history)}개, "
                                 f"로봇위치=({self.robot_x:.2f}, {self.robot_y:.2f})")
                                 
        except Exception as e:
            self.get_logger().error(f"위험도 맵 업데이트 오류: {str(e)}")
            traceback.print_exc()
            
    def create_occupancy_grid(self):
        """OccupancyGrid 메시지 생성"""
        grid = OccupancyGrid()
        
        # 헤더 설정
        grid.header = Header()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = self.map_frame
        
        # 맵 메타데이터 설정
        grid.info.resolution = self.map_resolution
        grid.info.width = self.map_width
        grid.info.height = self.map_height
        
        # 맵 원점 설정 (맵 중심)
        grid.info.origin.position.x = -self.map_width * self.map_resolution / 2.0
        grid.info.origin.position.y = -self.map_height * self.map_resolution / 2.0
        grid.info.origin.position.z = 0.0
        grid.info.origin.orientation.w = 1.0
        
        # 위험도 맵을 occupancy 데이터로 변환
        # 위험도가 높을수록 occupancy 값이 높음 (0-100)
        occupancy_data = []
        for y in range(self.map_height):
            for x in range(self.map_width):
                risk_value = self.risk_map[y, x]
                
                # 위험도를 occupancy 값으로 변환
                if risk_value > self.risk_threshold:
                    # 위험도가 높을수록 occupancy 값 증가
                    occupancy_value = min(100, int(risk_value * 100))
                else:
                    occupancy_value = 0
                    
                occupancy_data.append(occupancy_value)
                
        grid.data = occupancy_data
        return grid
        
    def create_risk_map_visualization(self):
        """위험도 맵 시각화 이미지 생성"""
        # 위험도 맵을 컬러 이미지로 변환
        viz_image = np.zeros((self.map_height, self.map_width, 3), dtype=np.uint8)
        
        # 위험도 레벨별 색상 매핑
        for y in range(self.map_height):
            for x in range(self.map_width):
                risk_value = self.risk_map[y, x]
                
                if risk_value > self.risk_threshold:
                    # 위험도에 따른 색상 결정
                    if risk_value > 0.7:
                        color = self.risk_colors["high"]  # 빨간색
                    elif risk_value > 0.4:
                        color = self.risk_colors["medium"]  # 노란색
                    else:
                        color = self.risk_colors["low"]  # 녹색
                        
                    # 위험도에 따른 투명도 조정
                    alpha = min(255, int(risk_value * 255))
                    viz_image[y, x] = [int(c * alpha / 255) for c in color]
                else:
                    # 배경색 (어두운 회색)
                    viz_image[y, x] = [20, 20, 20]
                    
        # 맵 그리드 추가
        grid_spacing = 10  # 1미터 간격
        for i in range(0, self.map_width, grid_spacing):
            cv2.line(viz_image, (i, 0), (i, self.map_height), (50, 50, 50), 1)
        for i in range(0, self.map_height, grid_spacing):
            cv2.line(viz_image, (0, i), (self.map_width, i), (50, 50, 50), 1)
            
        # 맵 중심점 표시
        cv2.circle(viz_image, (self.map_center_x, self.map_center_y), 3, (255, 255, 255), -1)
        
        # 로봇 위치 표시
        robot_map_x, robot_map_y = self.global_to_map_coordinates(self.robot_x, self.robot_y)
        if 0 <= robot_map_x < self.map_width and 0 <= robot_map_y < self.map_height:
            cv2.circle(viz_image, (robot_map_x, robot_map_y), 5, (0, 255, 255), -1)  # 노란색
            cv2.circle(viz_image, (robot_map_x, robot_map_y), 8, (0, 255, 255), 2)   # 노란색 테두리
            
        # 위험도 히스토리 표시
        for risk_obj in self.risk_history:
            if time.time() - risk_obj['timestamp'] < 30.0:  # 30초 이내
                x, y = risk_obj['map_x'], risk_obj['map_y']
                if 0 <= x < self.map_width and 0 <= y < self.map_height:
                    # 위험도 레벨에 따른 색상
                    color = self.risk_colors.get(risk_obj.get('risk_level', 'low'), (0, 255, 0))
                    cv2.circle(viz_image, (x, y), 2, color, -1)
                    
        # 정보 텍스트 추가
        max_risk = np.max(self.risk_map)
        avg_risk = np.mean(self.risk_map)
        risk_pixels = np.sum(self.risk_map > self.risk_threshold)
        
        info_text = f"Max: {max_risk:.3f} | Avg: {avg_risk:.3f} | Risk Pixels: {risk_pixels}"
        cv2.putText(viz_image, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 로봇 위치 정보
        robot_text = f"Robot: ({self.robot_x:.2f}, {self.robot_y:.2f})"
        cv2.putText(viz_image, robot_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 맵 크기 정보
        size_text = f"Map: {self.map_width*self.map_resolution:.1f}m x {self.map_height*self.map_resolution:.1f}m"
        cv2.putText(viz_image, size_text, (10, self.map_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return viz_image
        
    def broadcast_map_transform(self):
        """맵 프레임 TF 브로드캐스트"""
        try:
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = self.odom_frame
            t.child_frame_id = self.map_frame
            
            # 맵은 odom 프레임을 기준으로 고정
            t.transform.translation.x = 0.0
            t.transform.translation.y = 0.0
            t.transform.translation.z = 0.0
            t.transform.rotation.w = 1.0
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            
            self.tf_broadcaster.sendTransform(t)
            
        except Exception as e:
            self.get_logger().error(f"TF 브로드캐스트 오류: {str(e)}")
            
    def scan_callback(self, msg: LaserScan):
        """LiDAR 콜백"""
        self.latest_scan = msg

def main(args=None):
    rclpy.init(args=args)
    node = None
    
    try:
        node = RiskMapNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"노드 실행 오류: {str(e)}")
    finally:
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception as e:
            print(f"rclpy 종료 오류 (무시됨): {str(e)}")

if __name__ == '__main__':
    main()
 