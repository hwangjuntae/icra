#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
import threading
import queue
import time
import sys

# EfficientSAM 경로 추가
sys.path.append('/home/teus/ws/src/risk_nav/EfficientSAM')

# BLIP2 관련 import
try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    from PIL import Image as PILImage
    import torch
    BLIP2_AVAILABLE = True
except ImportError:
    BLIP2_AVAILABLE = False
    print("BLIP2 라이브러리가 없습니다. conda activate cam 후 설치하세요.")

# EfficientSAM 관련 import
try:
    from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
    from torchvision import transforms
    import torch
    import zipfile
    EFFICIENT_SAM_AVAILABLE = True
except ImportError:
    EFFICIENT_SAM_AVAILABLE = False
    print("EfficientSAM 라이브러리가 없습니다. conda activate cam 후 설치하세요.")

class CameraRGBProcessor(Node):
    def __init__(self):
        super().__init__('camera_rgb_processor')
        
        # 이미지 저장 폴더 설정
        self.save_dir = os.path.join(os.path.dirname(__file__), 'Camera_rgb')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # OpenCV 브리지 생성
        self.bridge = CvBridge()
        
        # 처리 카운터
        self.process_count = 0
        
        # 이미지 큐 (실시간 처리를 위한 큐)
        self.image_queue = queue.Queue(maxsize=5)
        
        # 결과 저장 경로
        self.latest_image_path = os.path.join(self.save_dir, "camera_rgb_latest.jpg")
        self.caption_path = os.path.join(self.save_dir, "latest_caption.txt")
        self.segmented_path = os.path.join(self.save_dir, "segmented_latest.jpg")
        
        # 모델 초기화
        self.init_models()
        
        # 처리 스레드 시작
        self.processing_thread = threading.Thread(target=self.process_images_worker, daemon=True)
        self.processing_thread.start()
        
        # /Camera/rgb 토픽 구독
        self.subscription = self.create_subscription(
            Image,
            '/Camera/rgb',
            self.image_callback,
            10)
        
        self.get_logger().info('Camera RGB Processor 노드가 시작되었습니다.')
        self.get_logger().info(f'이미지 저장 폴더: {self.save_dir}')
        self.get_logger().info(f'BLIP2 사용 가능: {BLIP2_AVAILABLE}')
        self.get_logger().info(f'EfficientSAM 사용 가능: {EFFICIENT_SAM_AVAILABLE}')
        self.get_logger().info('실시간 이미지 처리를 시작합니다...')
    
    def init_models(self):
        """BLIP2과 EfficientSAM 모델 초기화"""
        self.get_logger().info('모델들을 초기화하는 중...')
        
        # BLIP2 모델 초기화
        if BLIP2_AVAILABLE:
            try:
                self.get_logger().info('BLIP2 모델 로드 중...')
                self.blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
                self.blip2_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
                
                # GPU 사용 가능하면 GPU로 이동
                if torch.cuda.is_available():
                    self.blip2_model = self.blip2_model.to("cuda")
                    self.get_logger().info('BLIP2 모델 로드 완료 (GPU)')
                else:
                    self.get_logger().info('BLIP2 모델 로드 완료 (CPU)')
                    
            except Exception as e:
                self.get_logger().error(f'BLIP2 모델 로드 실패: {e}')
                self.blip2_processor = None
                self.blip2_model = None
        
        # EfficientSAM 모델 초기화
        if EFFICIENT_SAM_AVAILABLE:
            try:
                self.get_logger().info('EfficientSAM 모델 로드 중...')
                
                # EfficientSAM 경로 설정
                efficient_sam_path = "/home/teus/ws/src/risk_nav/EfficientSAM"
                os.chdir(efficient_sam_path)
                
                # EfficientSAM-S 모델 압축 해제 (필요시)
                weights_dir = os.path.join(efficient_sam_path, "weights")
                zip_path = os.path.join(weights_dir, "efficient_sam_vits.pt.zip")
                model_path = os.path.join(weights_dir, "efficient_sam_vits.pt")
                
                if os.path.exists(zip_path) and not os.path.exists(model_path):
                    self.get_logger().info('EfficientSAM-S 모델 압축 해제 중...')
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(weights_dir)
                
                # EfficientSAM-Ti 모델 사용 (더 빠름)
                self.efficient_sam_model = build_efficient_sam_vitt()
                
                # 원래 경로로 복원
                os.chdir("/home/teus/ws/src/risk_nav/src")
                
                self.get_logger().info('EfficientSAM 모델 로드 완료')
                
            except Exception as e:
                self.get_logger().error(f'EfficientSAM 모델 로드 실패: {e}')
                self.efficient_sam_model = None
    
    def image_callback(self, msg):
        """카메라 이미지 데이터를 받아오는 콜백 함수"""
        try:
            # ROS Image 메시지를 OpenCV 이미지로 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 큐에 이미지 추가 (큐가 가득 차면 가장 오래된 이미지 제거)
            if self.image_queue.full():
                try:
                    self.image_queue.get_nowait()
                except queue.Empty:
                    pass
            
            # 새 이미지를 큐에 추가
            self.image_queue.put((cv_image, msg.header.stamp, time.time()))
            
        except Exception as e:
            self.get_logger().error(f'이미지 콜백 처리 중 오류 발생: {str(e)}')
    
    def process_images_worker(self):
        """별도 스레드에서 이미지 처리"""
        while True:
            try:
                # 큐에서 이미지 가져오기 (타임아웃 1초)
                cv_image, timestamp, receive_time = self.image_queue.get(timeout=1.0)
                
                # 처리 시작 시간
                start_time = time.time()
                
                # 카운터 증가
                self.process_count += 1
                
                # 기본 이미지 저장
                cv2.imwrite(self.latest_image_path, cv_image)
                
                # BLIP2로 이미지 캡셔닝
                caption = self.generate_caption(cv_image)
                
                # EfficientSAM으로 세그멘테이션
                segmented_image = self.segment_image(cv_image)
                
                # 처리 시간 계산
                process_time = time.time() - start_time
                
                # 결과 로그
                self.get_logger().info(f'=== 처리 완료 #{self.process_count} ===')
                self.get_logger().info(f'처리 시간: {process_time:.2f}초')
                self.get_logger().info(f'캡션: {caption}')
                self.get_logger().info(f'세그멘테이션: {"완료" if segmented_image is not None else "실패"}')
                
                # 큐 완료 표시
                self.image_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'이미지 처리 중 오류: {str(e)}')
    
    def generate_caption(self, cv_image):
        """BLIP2를 사용한 이미지 캡셔닝"""
        if not BLIP2_AVAILABLE or self.blip2_processor is None:
            return "BLIP2 모델을 사용할 수 없습니다"
        
        try:
            # OpenCV 이미지를 PIL 이미지로 변환
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb_image)
            
            # BLIP2 처리
            inputs = self.blip2_processor(pil_image, return_tensors="pt")
            
            # GPU 사용 가능하면 GPU로 이동
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # 캡션 생성
            with torch.no_grad():
                generated_ids = self.blip2_model.generate(**inputs, max_length=50)
                caption = self.blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # 캡션 저장
            with open(self.caption_path, 'w', encoding='utf-8') as f:
                f.write(caption)
            
            return caption
            
        except Exception as e:
            self.get_logger().error(f'BLIP2 처리 중 오류: {str(e)}')
            return "캡션 생성 실패"
    
    def segment_image(self, cv_image):
        """EfficientSAM을 사용한 이미지 세그멘테이션"""
        if not EFFICIENT_SAM_AVAILABLE or self.efficient_sam_model is None:
            return None
        
        try:
            # OpenCV 이미지를 PIL 이미지로 변환 후 텐서로 변환
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb_image)
            image_tensor = transforms.ToTensor()(pil_image)
            
            # 이미지 중앙 포인트 설정
            h, w = cv_image.shape[:2]
            center_x, center_y = w // 2, h // 2
            
            # EfficientSAM 입력 형태로 변환
            input_points = torch.tensor([[[[center_x, center_y]]]])
            input_labels = torch.tensor([[[1]]])
            
            # 세그멘테이션 실행
            with torch.no_grad():
                predicted_logits, predicted_iou = self.efficient_sam_model(
                    image_tensor[None, ...],
                    input_points,
                    input_labels,
                )
            
            # 가장 좋은 마스크 선택
            sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
            predicted_logits = torch.take_along_dim(
                predicted_logits, sorted_ids[..., None, None], dim=2
            )
            
            # 마스크 생성
            mask = torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy()
            
            # 세그멘테이션 결과 시각화
            segmented_image = cv_image.copy()
            
            # 마스크 적용 (초록색 오버레이)
            overlay = segmented_image.copy()
            overlay[mask] = [0, 255, 0]  # 초록색
            segmented_image = cv2.addWeighted(segmented_image, 0.7, overlay, 0.3, 0)
            
            # 세그멘테이션 이미지 저장
            cv2.imwrite(self.segmented_path, segmented_image)
            
            return segmented_image
            
        except Exception as e:
            self.get_logger().error(f'EfficientSAM 처리 중 오류: {str(e)}')
            return None

def main(args=None):
    # ROS2 초기화
    rclpy.init(args=args)
    
    # 노드 생성
    node = CameraRGBProcessor()
    
    try:
        # 노드 실행 (무한 루프)
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('사용자에 의해 종료됩니다.')
        node.get_logger().info(f'총 {node.process_count}개의 이미지가 처리되었습니다.')
    finally:
        # 정리 작업
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 