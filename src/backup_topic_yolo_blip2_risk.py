#!/usr/bin/env python3
"""
YOLO + BLIP2 위험도 평가 ROS2 노드
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
import cv2
import numpy as np
import json
import time
import traceback
import os
from pathlib import Path

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
        
        # 10-15Hz 제어를 위한 변수들
        self.target_fps = 12.0  # 10-15Hz 범위의 중간값으로 설정
        self.frame_interval = 1.0 / self.target_fps  # 0.083 seconds (83.3ms)
        self.last_process_time = 0.0
        self.latest_image = None
        
        # 해상도 설정 (처리 속도 향상을 위해)
        self.processing_scale = 0.6  # 60% 크기로 다운샘플링
        self.min_processing_width = 320  # 최소 처리 너비
        self.min_processing_height = 240  # 최소 처리 높이
        
        # 성능 모니터링
        self.processing_times = []
        self.max_time_samples = 10
        
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
        
        self.get_logger().info("YOLO + BLIP2 위험도 평가 노드 시작")
        self.get_logger().info("구독 토픽: /Camera/rgb")
        self.get_logger().info("발행 토픽: /risk_assessment/image (sensor_msgs/Image)")
        self.get_logger().info(f"목표 처리 주파수: {self.target_fps}Hz ({self.frame_interval*1000:.1f}ms 간격)")
        self.get_logger().info(f"처리 해상도 스케일: {self.processing_scale:.1f} (최소: {self.min_processing_width}x{self.min_processing_height})")
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
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.blip_model.to(self.device)
                self.get_logger().info(f"BLIP2 모델 로드 완료 (디바이스: {self.device})")
            else:
                self.blip_processor = None
                self.blip_model = None
                
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
                    
                    self.get_logger().info(f"위험도 평가 결과: {len(risk_results.get('detections', []))}개 객체 | "
                                         f"처리시간: {processing_time:.1f}ms | 실제 FPS: {actual_fps:.1f} | "
                                         f"처리 해상도: {processed_image.shape[1]}x{processed_image.shape[0]}")
                    
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
                if self.processing_scale > 0.3:  # 최소 30%까지만 줄임
                    self.processing_scale *= 0.9
                    self.get_logger().info(f"처리 스케일 조정: {self.processing_scale:.2f}")
            
    def image_callback(self, msg):
        """이미지 콜백 함수 - 최신 이미지만 저장"""
        try:
            if self.bridge is None:
                self.get_logger().error("cv_bridge를 사용할 수 없습니다.")
                return
                
            # ROS 이미지를 OpenCV 이미지로 변환 (안전한 방법)
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception as bridge_error:
                self.get_logger().error(f"cv_bridge 변환 오류: {str(bridge_error)}")
                # 대안적인 방법으로 이미지 변환 시도
                cv_image = self.manual_image_conversion(msg)
                if cv_image is None:
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
            
            label = f"{class_name} {confidence:.2f}"
            risk_text = f"Risk: {risk_score:.1f} ({risk_level})"
            
            # 텍스트 배경 박스 크기 계산
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            (risk_w, risk_h), _ = cv2.getTextSize(risk_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            
            # 텍스트 배경 박스 그리기
            cv2.rectangle(vis_image, (x1, y1 - label_h - risk_h - 10), 
                         (x1 + max(label_w, risk_w) + 10, y1), color, -1)
            
            # 텍스트 그리기
            cv2.putText(vis_image, label, (x1 + 5, y1 - risk_h - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(vis_image, risk_text, (x1 + 5, y1 - 5), 
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
        """위험도 평가 수행"""
        results = {
            "timestamp": time.time(),
            "detections": [],
            "overall_risk_score": 0.0,
            "risk_level": "low",
            "scene_description": "장면 분석 중..."
        }
        
        try:
            # YOLO 객체 탐지
            if self.yolo_model is not None:
                yolo_results = self.yolo_model(image)
                
                if len(yolo_results) > 0 and yolo_results[0].boxes is not None:
                    total_risk = 0.0
                    detection_count = 0
                    
                    for detection in yolo_results[0].boxes:
                        # 바운딩 박스 정보 추출 (다운샘플된 이미지 기준)
                        x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
                        confidence = detection.conf[0].cpu().numpy()
                        class_id = int(detection.cls[0].cpu().numpy())
                        class_name = self.yolo_model.names[class_id]
                        
                        # 바운딩 박스를 원본 크기로 스케일링
                        if scale_info is not None:
                            x1 *= scale_info['scale_x']
                            y1 *= scale_info['scale_y']
                            x2 *= scale_info['scale_x']
                            y2 *= scale_info['scale_y']
                        
                        # 위험도 계산
                        risk_score = self.calculate_risk_score(class_name, confidence)
                        risk_level = self.get_risk_level(risk_score)
                        
                        # 객체 영역 추출 및 BLIP2 설명 (다운샘플된 이미지에서)
                        obj_description = self.get_object_description(image, 
                                                                    x1 / scale_info['scale_x'] if scale_info else x1,
                                                                    y1 / scale_info['scale_y'] if scale_info else y1,
                                                                    x2 / scale_info['scale_x'] if scale_info else x2,
                                                                    y2 / scale_info['scale_y'] if scale_info else y2,
                                                                    class_name)
                        
                        detection_result = {
                            "class_name": class_name,
                            "confidence": float(confidence),
                            "bbox": {
                                "x1": float(x1),
                                "y1": float(y1), 
                                "x2": float(x2),
                                "y2": float(y2)
                            },
                            "risk_score": float(risk_score),
                            "risk_level": risk_level,
                            "description": obj_description
                        }
                        
                        results["detections"].append(detection_result)
                        total_risk += risk_score
                        detection_count += 1
                    
                    # 전체 위험도 계산
                    if detection_count > 0:
                        results["overall_risk_score"] = total_risk / detection_count
                        results["risk_level"] = self.get_risk_level(results["overall_risk_score"])
            
            # BLIP2로 전체 장면 설명 (다운샘플된 이미지로)
            if self.blip_model is not None:
                results["scene_description"] = self.get_scene_description(image)
                
        except Exception as e:
            self.get_logger().error(f"위험도 평가 오류: {str(e)}")
            
        return results
        
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
        """객체 설명 생성"""
        if self.blip_model is None:
            return f"{class_name} 객체"
            
        try:
            # 객체 영역 추출
            obj_image = image[int(y1):int(y2), int(x1):int(x2)]
            if obj_image.size == 0:
                return f"{class_name} 객체"
                
            # BGR을 RGB로 변환
            obj_rgb = cv2.cvtColor(obj_image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(obj_rgb)
            
            # BLIP2로 설명 생성
            inputs = self.blip_processor(pil_image, return_tensors="pt").to(self.device)
            out = self.blip_model.generate(**inputs, max_length=50)
            description = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            return description
            
        except Exception as e:
            self.get_logger().error(f"객체 설명 생성 실패: {str(e)}")
            return f"{class_name} 객체"
            
    def get_scene_description(self, image):
        """전체 장면 설명 생성"""
        if self.blip_model is None:
            return "장면 설명을 생성할 수 없습니다."
            
        try:
            # BGR을 RGB로 변환
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb_image)
            
            # BLIP2로 장면 설명 생성
            inputs = self.blip_processor(pil_image, return_tensors="pt").to(self.device)
            out = self.blip_model.generate(**inputs, max_length=100)
            description = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            return description
            
        except Exception as e:
            self.get_logger().error(f"장면 설명 생성 실패: {str(e)}")
            return "장면 설명을 생성할 수 없습니다."

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = YOLOBlip2RiskNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"노드 실행 오류: {str(e)}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
