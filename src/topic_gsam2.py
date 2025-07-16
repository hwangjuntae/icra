#!/usr/bin/env python3

import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert, nms
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
from PIL import Image, ImageEnhance

# ROS2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
from std_msgs.msg import String
import threading
import time

# Add Grounded-SAM-2 path and change working directory
grounded_sam2_path = '/root/ws/src/risk_nav/src/Grounded-SAM-2'
sys.path.append(grounded_sam2_path)
# Note: We'll temporarily change directory only when needed for model loading

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

class RealTimeRiskGSAM2(Node):
    def __init__(self):
        super().__init__('realtime_risk_gsam2')
        
        # ROS2 설정
        self.bridge = CvBridge()
        self.image_subscriber = self.create_subscription(
            RosImage,
            '/Camera/rgb',
            self.image_callback,
            10
        )
        
        # 결과 발행을 위한 publisher
        self.result_publisher = self.create_publisher(
            String,
            '/risk_assessment/result',
            10
        )
        
        # 위험도 분석 이미지 발행을 위한 publisher
        self.image_publisher = self.create_publisher(
            RosImage,
            '/risk_assessment/image',
            10
        )
        
        # 처리 중인지 확인하기 위한 플래그
        self.processing = False
        self.last_process_time = time.time()
        self.process_interval = 0.5  # 0.5초마다 처리 (부하 조절)
        
        # 모델 초기화
        self.get_logger().info("모델 로딩 중...")
        self.initialize_models()
        self.get_logger().info("실시간 위험 분석 시스템이 시작되었습니다.")
        
        # 설정 변수들
        self.setup_configuration()
        
        # 출력 디렉토리 설정
        self.output_dir = "/root/ws/src/risk_nav/src/sample/outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def setup_configuration(self):
        """기존 risk_gsam2.py의 설정들을 가져옴"""
        # Object prompts
        self.object_prompts = {
            "basic_furniture": "office chair. desk. table. cabinet. shelf. drawer.",
            "electronics": "computer monitor. laptop computer. desktop computer. keyboard. computer mouse. phone. printer. scanner.",
            "lighting": "ceiling light. desk lamp. floor lamp. light fixture. LED panel.",
            "safety_equipment": "fire extinguisher. emergency exit sign. first aid kit. smoke detector.",
            "structure": "ceiling. floor. wall. window. door. column. beam.",
            "hazards": "electrical cable. power cord. extension cord. loose wire. staircase. stairs. step. edge. corner. glass panel. sharp edge.",
            "storage": "cardboard box. storage box. bag. backpack. briefcase. folder. filing cabinet.",
            "consumables": "paper. book. document. pen. pencil. cup. mug. bottle. trash can."
        }
        
        # Combine all prompts
        self.all_prompts = " ".join(self.object_prompts.values())
        
        # Detection thresholds
        self.detection_thresholds = [
            {"box_threshold": 0.25, "text_threshold": 0.20},
            {"box_threshold": 0.35, "text_threshold": 0.25},
            {"box_threshold": 0.45, "text_threshold": 0.30},
        ]
        
        # Safety categories
        self.safety_categories = {
            "floor": {
                "objects": ["floor", "ground", "surface"],
                "color": (139, 69, 19),
                "alpha": 0.5,
                "description": "Floor Area",
                "base_risk": 5
            },
            "ceiling": {
                "objects": ["ceiling", "roof", "top"],
                "color": (135, 206, 235),
                "alpha": 0.5,
                "description": "Ceiling Area",
                "base_risk": 2
            },
            "safe": {
                "objects": ["computer", "monitor", "laptop", "keyboard", "mouse", "chair", "desk", "table", "lamp", "light", "phone", "book", "paper", "pen", "cup", "bottle", "fire extinguisher", "emergency exit", "first aid", "smoke detector", "wall", "window"],
                "color": (0, 255, 0),
                "alpha": 0.3,
                "description": "Safe Objects",
                "base_risk": 1
            },
            "dynamic": {
                "objects": ["door", "corner", "edge", "entrance", "exit", "opening", "passage", "blind spot", "intersection", "junction", "turning", "column", "pillar", "beam"],
                "color": (255, 255, 0),
                "alpha": 0.4,
                "description": "Dynamic Environment",
                "base_risk": 15
            },
            "caution": {
                "objects": ["glass", "panel", "box", "bag", "backpack", "briefcase", "cabinet", "shelf", "drawer", "filing", "step"],
                "color": (255, 165, 0),
                "alpha": 0.4,
                "description": "Caution Objects",
                "base_risk": 10
            },
            "danger": {
                "objects": ["cable", "wire", "cord", "stairs", "staircase", "sharp", "loose", "electrical", "power", "extension", "obstacle", "hazard"],
                "color": (255, 0, 0),
                "alpha": 0.5,
                "description": "Danger Objects",
                "base_risk": 25
            }
        }
        
        # Risk levels
        self.risk_levels = {
            "SAFE": {"range": (0, 10), "color": (0, 255, 0), "description": "Safe"},
            "LOW": {"range": (11, 30), "color": (255, 255, 0), "description": "Low Risk"},
            "MEDIUM": {"range": (31, 60), "color": (255, 165, 0), "description": "Medium Risk"},
            "HIGH": {"range": (61, 85), "color": (255, 100, 0), "description": "High Risk"},
            "CRITICAL": {"range": (86, 100), "color": (255, 0, 0), "description": "Critical Risk"}
        }
        
    def initialize_models(self):
        """모델들을 초기화"""
        # 현재 디렉토리 저장
        original_dir = os.getcwd()
        
        try:
            # Grounded-SAM-2 디렉토리로 이동
            os.chdir(grounded_sam2_path)
            
            # Device setup
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.get_logger().info(f"디바이스: {self.device}")
            
            # Model paths (relative to Grounded-SAM-2 directory)
            sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
            sam2_model_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
            grounding_dino_config = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
            grounding_dino_checkpoint = "gdino_checkpoints/groundingdino_swint_ogc.pth"
            
            # Load SAM2 model
            self.sam2_model = build_sam2(sam2_model_config, sam2_checkpoint, device=self.device)
            self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
            
            # Load Grounding DINO model
            self.grounding_model = load_model(grounding_dino_config, grounding_dino_checkpoint, device=self.device)
            
        finally:
            # 원래 디렉토리로 복원
            os.chdir(original_dir)
    
    def image_callback(self, msg):
        """이미지 토픽 콜백 함수"""
        current_time = time.time()
        
        # 처리 중이거나 간격이 짧으면 건너뛰기
        if self.processing or (current_time - self.last_process_time) < self.process_interval:
            return
            
        try:
            # ROS 이미지를 OpenCV 이미지로 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 별도 스레드에서 처리 (메인 스레드 블록 방지)
            processing_thread = threading.Thread(
                target=self.process_image,
                args=(cv_image, current_time)
            )
            processing_thread.daemon = True
            processing_thread.start()
            
        except Exception as e:
            self.get_logger().error(f"이미지 콜백 오류: {str(e)}")
    
    def process_image(self, cv_image, timestamp):
        """이미지 처리 함수"""
        self.processing = True
        self.last_process_time = timestamp
        
        try:
            self.get_logger().info("이미지 처리 시작...")
            
            # OpenCV 이미지를 RGB로 변환
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # 이미지 전처리
            processed_image = self.preprocess_image(rgb_image)
            
            # Grounded-SAM-2 디렉토리로 이동하여 처리
            original_dir = os.getcwd()
            os.chdir(grounded_sam2_path)
            
            try:
                # 객체 탐지 (이 함수 내에서 load_image 사용)
                boxes, confidences, labels = self.multi_threshold_detection(processed_image)
                
                if len(boxes) == 0:
                    self.get_logger().warn("객체가 탐지되지 않았습니다.")
                    return
                
                # SAM2 predictor에 이미지 설정 (원본 이미지 사용)
                self.sam2_predictor.set_image(processed_image)
                
                # 박스 좌표 변환
                h, w, _ = processed_image.shape
                boxes = boxes * torch.Tensor([w, h, w, h])
                input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
                
                # NMS 적용
                input_boxes, confidences, labels = self.apply_nms(input_boxes, confidences, labels)
                
                # 신뢰도 필터링
                input_boxes, confidences, labels = self.filter_low_confidence(input_boxes, confidences, labels)
                
                if len(input_boxes) == 0:
                    self.get_logger().warn("필터링 후 객체가 없습니다.")
                    return
                
                # numpy로 변환
                input_boxes = input_boxes.numpy()
                
                # 세그멘테이션
                masks, scores, logits = self.sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )
                
                # 결과 시각화 및 위험도 계산
                result_image, risk_scores, risk_levels = self.create_colored_mask_overlay(
                    processed_image, masks, labels, confidences, input_boxes
                )
                
                # 결과 저장
                self.save_results(result_image, labels, risk_scores, risk_levels, input_boxes, confidences, masks)
                
                # 위험도 분석 결과 발행
                self.publish_risk_assessment(labels, risk_scores, risk_levels)
                
                self.get_logger().info(f"이미지 처리 완료: {len(labels)}개 객체 탐지")
                
            finally:
                os.chdir(original_dir)
                
        except Exception as e:
            self.get_logger().error(f"이미지 처리 오류: {str(e)}")
        finally:
            self.processing = False
    
    def preprocess_image(self, image):
        """이미지 전처리 (기존 함수 재활용)"""
        # PIL 이미지로 변환
        pil_image = Image.fromarray(image)
        
        # 이미지 향상
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.2)
        
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(1.1)
        
        # 크기 조정
        max_size = 1024
        if max(pil_image.size) > max_size:
            ratio = max_size / max(pil_image.size)
            new_size = tuple(int(dim * ratio) for dim in pil_image.size)
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        
        # numpy 배열로 변환
        return np.array(pil_image)
    
    def multi_threshold_detection(self, image):
        """다중 임계값 탐지"""
        all_boxes = []
        all_confidences = []
        all_labels = []
        
        # 임시 파일로 저장하여 load_image 함수 사용
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            # PIL 이미지로 변환 후 저장
            pil_image = Image.fromarray(image)
            pil_image.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            # load_image 함수로 올바른 형태로 로드
            image_source, processed_image = load_image(tmp_path)
            
            for i, thresholds in enumerate(self.detection_thresholds):
                try:
                    boxes, confidences, labels = predict(
                        model=self.grounding_model,
                        image=processed_image,
                        caption=self.all_prompts,
                        box_threshold=thresholds['box_threshold'],
                        text_threshold=thresholds['text_threshold'],
                        device=self.device
                    )
                    
                    # 결과가 있으면 추가
                    if len(boxes) > 0:
                        all_boxes.append(boxes)
                        all_confidences.append(confidences)
                        all_labels.extend(labels)
                        
                except Exception as e:
                    self.get_logger().error(f"탐지 실행 오류 (임계값 {i}): {str(e)}")
                    continue
                    
        except Exception as e:
            self.get_logger().error(f"이미지 로딩 오류: {str(e)}")
            
        finally:
            # 임시 파일 삭제
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        if not all_boxes:
            return torch.empty((0, 4)), torch.empty(0), []
        
        # 결과 결합
        combined_boxes = torch.cat(all_boxes, dim=0)
        combined_confidences = torch.cat(all_confidences, dim=0)
        
        return combined_boxes, combined_confidences, all_labels
    
    def apply_nms(self, boxes, confidences, labels, iou_threshold=0.5):
        """NMS 적용"""
        if len(boxes) == 0:
            return boxes, confidences, labels
        
        # 라벨별로 NMS 적용
        unique_labels = list(set(labels))
        final_boxes = []
        final_confidences = []
        final_labels = []
        
        for label in unique_labels:
            label_indices = [i for i, l in enumerate(labels) if l == label]
            
            if len(label_indices) == 0:
                continue
                
            label_boxes = boxes[label_indices]
            label_confidences = confidences[label_indices]
            
            if len(label_boxes) > 1:
                keep_indices = nms(label_boxes, label_confidences, iou_threshold)
                keep_indices = keep_indices.cpu().numpy()
                
                final_boxes.extend(label_boxes[keep_indices])
                final_confidences.extend(label_confidences[keep_indices])
                final_labels.extend([label] * len(keep_indices))
            else:
                final_boxes.extend(label_boxes)
                final_confidences.extend(label_confidences)
                final_labels.extend([label])
        
        if final_boxes:
            final_boxes = torch.stack(final_boxes)
            final_confidences = torch.stack(final_confidences)
        else:
            final_boxes = torch.empty((0, 4))
            final_confidences = torch.empty(0)
            final_labels = []
        
        return final_boxes, final_confidences, final_labels
    
    def filter_low_confidence(self, boxes, confidences, labels, min_confidence=0.3):
        """낮은 신뢰도 필터링"""
        if len(boxes) == 0:
            return boxes, confidences, labels
        
        high_conf_indices = confidences >= min_confidence
        
        filtered_boxes = boxes[high_conf_indices]
        filtered_confidences = confidences[high_conf_indices]
        filtered_labels = [labels[i] for i in range(len(labels)) if high_conf_indices[i]]
        
        return filtered_boxes, filtered_confidences, filtered_labels
    
    def categorize_object(self, label):
        """객체 분류"""
        label_lower = label.lower()
        
        # 바닥 분류 우선
        if any(keyword in label_lower for keyword in ["floor", "ground", "surface"]):
            return "floor"
        
        # 천장 분류 우선
        if any(keyword in label_lower for keyword in ["ceiling", "roof", "top"]):
            return "ceiling"
        
        # 동적 환경 분류
        dynamic_keywords = ["door", "corner", "edge", "entrance", "exit", "opening", "passage", "intersection", "junction", "turning", "column", "pillar", "beam"]
        if any(keyword in label_lower for keyword in dynamic_keywords):
            return "dynamic"
        
        # 나머지 카테고리 분류
        for category, info in self.safety_categories.items():
            if category in ["floor", "ceiling", "dynamic"]:
                continue
            
            for obj_keyword in info["objects"]:
                if obj_keyword in label_lower or label_lower in obj_keyword:
                    return category
        
        return "safe"
    
    def calculate_risk_score(self, category, confidence, box, image_shape):
        """위험도 점수 계산"""
        base_risk = self.safety_categories[category]["base_risk"]
        confidence_weight = float(confidence) * 20
        
        if isinstance(box, torch.Tensor):
            box = box.cpu().numpy()
        
        x1, y1, x2, y2 = box
        object_area = (x2 - x1) * (y2 - y1)
        image_area = image_shape[0] * image_shape[1]
        size_ratio = object_area / image_area
        
        if category in ["danger", "caution"]:
            size_weight = size_ratio * 30
        elif category == "dynamic":
            size_weight = size_ratio * 20
        else:
            size_weight = size_ratio * 5
        
        special_multiplier = 1.0
        if category == "danger":
            special_multiplier = 1.5
        elif category == "dynamic":
            special_multiplier = 1.2
        
        risk_score = (base_risk + confidence_weight + size_weight) * special_multiplier
        return max(0, min(100, float(risk_score)))
    
    def get_risk_level(self, risk_score):
        """위험도 레벨 반환"""
        for level, info in self.risk_levels.items():
            if info["range"][0] <= risk_score <= info["range"][1]:
                return level
        return "SAFE"
    
    def create_colored_mask_overlay(self, image, masks, labels, confidences, boxes):
        """컬러 마스크 오버레이 생성"""
        overlay = image.copy()
        
        risk_scores = []
        risk_levels = []
        
        for i, (mask, label, conf, box) in enumerate(zip(masks, labels, confidences, boxes)):
            category = self.categorize_object(label)
            risk_score = self.calculate_risk_score(category, conf, box, image.shape)
            risk_level = self.get_risk_level(risk_score)
            
            risk_scores.append(risk_score)
            risk_levels.append(risk_level)
            
            # 시각화 파라미터
            intensity = min(1.0, float(risk_score) / 100.0)
            border_thickness = int(2 + (float(risk_score) / 100.0) * 6)
            alpha = 0.3 + (float(risk_score) / 100.0) * 0.4
            
            # 기본 카테고리 색상
            base_color = self.safety_categories[category]["color"]
            risk_color = self.risk_levels[risk_level]["color"]
            
            # 마스크 적용 (차원 확인 및 안전한 처리)
            try:
                # 마스크 차원 확인 및 올바른 형태로 변환
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                
                # 마스크 차원 처리
                if mask.ndim == 3:
                    if mask.shape[0] == 1:
                        mask_2d = mask[0]  # (1, H, W) -> (H, W)
                    else:
                        # 다중 마스크인 경우 첫 번째 것만 사용
                        mask_2d = mask[0] if mask.shape[0] > 0 else mask
                elif mask.ndim == 2:
                    mask_2d = mask     # 이미 (H, W) 형태
                else:
                    self.get_logger().error(f"예상치 못한 마스크 차원: {mask.shape}")
                    continue
                
                # 마스크와 이미지 크기 확인 및 조정
                target_shape = image.shape[:2]  # (H, W)
                if mask_2d.shape != target_shape:
                    self.get_logger().warn(f"마스크 크기 불일치: 마스크 {mask_2d.shape}, 이미지 {target_shape}")
                    # 크기 조정
                    mask_2d = cv2.resize(mask_2d.astype(np.float32), 
                                       (target_shape[1], target_shape[0]), 
                                       interpolation=cv2.INTER_NEAREST)
                    mask_2d = mask_2d > 0.5  # 이진화
                
                # boolean 마스크로 변환
                mask_bool = mask_2d.astype(bool)
                
                # 안전한 마스크 적용
                if mask_bool.shape == overlay.shape[:2]:
                    overlay[mask_bool] = overlay[mask_bool] * (1 - alpha) + np.array(base_color) * alpha
                else:
                    self.get_logger().error(f"최종 마스크 크기 불일치: {mask_bool.shape} vs {overlay.shape[:2]}")
                    continue
                
            except Exception as e:
                self.get_logger().error(f"마스크 처리 오류: {str(e)}")
                continue
            
            # 위험도 기반 경계선
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), risk_color, border_thickness)
            
            # 라벨 텍스트
            category_names = {
                "floor": "FLOOR", "ceiling": "CEILING", "safe": "SAFE",
                "dynamic": "DYNAMIC", "caution": "CAUTION", "danger": "DANGER"
            }
            
            risk_names = {
                "SAFE": "SAFE", "LOW": "LOW", "MEDIUM": "MED",
                "HIGH": "HIGH", "CRITICAL": "CRIT"
            }
            
            category_name = category_names.get(category, "UNKNOWN")
            risk_name = risk_names.get(risk_level, "UNKNOWN")
            
            text = f"{category_name}: {label}"
            risk_text = f"{risk_name} ({risk_score:.1f})"
            
            # 텍스트 색상
            if category in ["danger", "floor"]:
                text_color = (255, 255, 255)
            else:
                text_color = (0, 0, 0)
            
            # 텍스트 배경
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            (risk_width, risk_height), _ = cv2.getTextSize(risk_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            max_width = max(text_width, risk_width)
            total_height = text_height + risk_height + 20
            
            # 배경 사각형
            cv2.rectangle(overlay, (x1, y1-total_height-10), (x1+max_width+20, y1), risk_color, -1)
            
            # 텍스트 그리기
            cv2.putText(overlay, text, (x1+5, y1-risk_height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            cv2.putText(overlay, risk_text, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        
        return overlay, risk_scores, risk_levels
    
    def save_results(self, result_image, labels, risk_scores, risk_levels, boxes, confidences, masks):
        """결과 저장 (기존 파일 삭제 후 최신 파일만 유지)"""
        import glob
        
        # 기존 실시간 분석 파일들 삭제
        old_files = glob.glob(os.path.join(self.output_dir, "realtime_risk_assessment_*"))
        for old_file in old_files:
            try:
                os.remove(old_file)
            except OSError:
                pass
        
        timestamp = int(time.time())
        
        # 이미지 저장
        output_path = os.path.join(self.output_dir, f"realtime_risk_assessment_{timestamp}.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        
        # 결과 이미지를 ROS 토픽으로 발행
        self.publish_result_image(result_image)
        
        # JSON 결과 저장
        results = {
            "timestamp": timestamp,
            "total_objects": len(labels),
            "average_risk_score": float(np.mean(risk_scores)) if risk_scores else 0,
            "risk_distribution": {level: 0 for level in self.risk_levels.keys()},
            "detailed_results": []
        }
        
        # 위험도 분포 계산
        for level in risk_levels:
            results["risk_distribution"][level] += 1
        
        # 상세 결과
        for i, (box, conf, label, risk_score, risk_level) in enumerate(zip(boxes, confidences, labels, risk_scores, risk_levels)):
            category = self.categorize_object(label)
            results["detailed_results"].append({
                "id": i,
                "label": label,
                "confidence": float(conf),
                "category": category,
                "risk_score": float(risk_score),
                "risk_level": risk_level,
                "bbox": box.tolist()
            })
        
        # JSON 파일 저장 (기존 JSON 파일도 삭제됨 - 위에서 glob으로 처리)
        json_path = os.path.join(self.output_dir, f"realtime_risk_assessment_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    def publish_risk_assessment(self, labels, risk_scores, risk_levels):
        """위험도 분석 결과 발행"""
        if not risk_scores:
            return
        
        # 위험도 요약 생성
        avg_risk = np.mean(risk_scores)
        max_risk = max(risk_scores)
        critical_count = sum(1 for level in risk_levels if level == "CRITICAL")
        high_count = sum(1 for level in risk_levels if level == "HIGH")
        
        # 결과 메시지 생성
        result_msg = {
            "timestamp": time.time(),
            "total_objects": len(labels),
            "average_risk_score": float(avg_risk),
            "max_risk_score": float(max_risk),
            "critical_objects": critical_count,
            "high_risk_objects": high_count,
            "status": "CRITICAL" if critical_count > 0 else "HIGH" if high_count > 0 else "NORMAL"
        }
        
        # String 메시지로 발행
        msg = String()
        msg.data = json.dumps(result_msg)
        self.result_publisher.publish(msg)
        
        # 로그 출력
        self.get_logger().info(f"위험도 분석 완료: 평균 {avg_risk:.1f}, 최대 {max_risk:.1f}, 위험 객체 {critical_count + high_count}개")
    
    def publish_result_image(self, result_image):
        """위험도 분석 결과 이미지를 ROS 토픽으로 발행"""
        try:
            # numpy 배열 (RGB)을 ROS Image 메시지로 변환
            ros_image = self.bridge.cv2_to_imgmsg(result_image, encoding="rgb8")
            
            # 헤더 설정
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = "camera_frame"
            
            # 이미지 발행
            self.image_publisher.publish(ros_image)
            
        except Exception as e:
            self.get_logger().error(f"이미지 발행 오류: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = RealTimeRiskGSAM2()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"노드 실행 오류: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 