#!/usr/bin/env python3

import cv2
import os
import numpy as np
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
    print("✅ BLIP2 라이브러리 로드 성공")
except ImportError as e:
    BLIP2_AVAILABLE = False
    print(f"❌ BLIP2 라이브러리 로드 실패: {e}")

# EfficientSAM 관련 import
try:
    from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
    from torchvision import transforms
    import torch
    import zipfile
    EFFICIENT_SAM_AVAILABLE = True
    print("✅ EfficientSAM 라이브러리 로드 성공")
except ImportError as e:
    EFFICIENT_SAM_AVAILABLE = False
    print(f"❌ EfficientSAM 라이브러리 로드 실패: {e}")

class ImageProcessor:
    def __init__(self):
        self.image_dir = os.path.join(os.path.dirname(__file__), 'Camera_rgb')
        self.image_path = os.path.join(self.image_dir, 'camera_rgb_latest.jpg')
        
        # 결과 저장 경로
        self.caption_path = os.path.join(self.image_dir, 'latest_caption.txt')
        self.segmented_path = os.path.join(self.image_dir, 'segmented_latest.jpg')
        
        # 모델 초기화
        self.blip2_processor = None
        self.blip2_model = None
        self.efficient_sam_model = None
        
        print("🚀 이미지 처리 시스템 초기화 중...")
        self.init_models()
    
    def init_models(self):
        """BLIP2과 EfficientSAM 모델 초기화"""
        print("📥 모델들을 로드하는 중...")
        
        # BLIP2 모델 초기화
        if BLIP2_AVAILABLE:
            try:
                print("🔄 BLIP2 모델 로드 중...")
                self.blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
                self.blip2_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
                
                # GPU 사용 가능하면 GPU로 이동
                if torch.cuda.is_available():
                    self.blip2_model = self.blip2_model.to("cuda")
                    print("✅ BLIP2 모델 로드 완료 (GPU)")
                else:
                    print("✅ BLIP2 모델 로드 완료 (CPU)")
                    
            except Exception as e:
                print(f"❌ BLIP2 모델 로드 실패: {e}")
                self.blip2_processor = None
                self.blip2_model = None
        
        # EfficientSAM 모델 초기화
        if EFFICIENT_SAM_AVAILABLE:
            try:
                print("🔄 EfficientSAM 모델 로드 중...")
                
                # EfficientSAM 경로 설정
                efficient_sam_path = "/home/teus/ws/src/risk_nav/EfficientSAM"
                current_dir = os.getcwd()
                os.chdir(efficient_sam_path)
                
                # EfficientSAM-S 모델 압축 해제 (필요시)
                weights_dir = os.path.join(efficient_sam_path, "weights")
                zip_path = os.path.join(weights_dir, "efficient_sam_vits.pt.zip")
                model_path = os.path.join(weights_dir, "efficient_sam_vits.pt")
                
                if os.path.exists(zip_path) and not os.path.exists(model_path):
                    print("📦 EfficientSAM-S 모델 압축 해제 중...")
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(weights_dir)
                
                # EfficientSAM-Ti 모델 사용 (더 빠름)
                self.efficient_sam_model = build_efficient_sam_vitt()
                
                # 원래 경로로 복원
                os.chdir(current_dir)
                
                print("✅ EfficientSAM 모델 로드 완료")
                
            except Exception as e:
                print(f"❌ EfficientSAM 모델 로드 실패: {e}")
                self.efficient_sam_model = None
    
    def generate_caption(self, cv_image):
        """BLIP2를 사용한 이미지 캡셔닝"""
        if not BLIP2_AVAILABLE or self.blip2_processor is None:
            return "BLIP2 모델을 사용할 수 없습니다"
        
        try:
            print("🔍 BLIP2 캡셔닝 중...")
            
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
            
            print(f"📝 캡션 생성 완료: {caption}")
            return caption
            
        except Exception as e:
            print(f"❌ BLIP2 처리 중 오류: {str(e)}")
            return "캡션 생성 실패"
    
    def segment_image(self, cv_image):
        """EfficientSAM을 사용한 이미지 세그멘테이션"""
        if not EFFICIENT_SAM_AVAILABLE or self.efficient_sam_model is None:
            return None
        
        try:
            print("🎯 EfficientSAM 세그멘테이션 중...")
            
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
            
            print("🎨 세그멘테이션 완료")
            return segmented_image
            
        except Exception as e:
            print(f"❌ EfficientSAM 처리 중 오류: {str(e)}")
            return None
    
    def process_image(self):
        """저장된 이미지 처리"""
        if not os.path.exists(self.image_path):
            print(f"❌ 이미지 파일이 없습니다: {self.image_path}")
            return
        
        print(f"📸 이미지 로드 중: {self.image_path}")
        
        # 이미지 로드
        cv_image = cv2.imread(self.image_path)
        if cv_image is None:
            print("❌ 이미지 로드 실패")
            return
        
        print(f"📏 이미지 크기: {cv_image.shape}")
        
        # 처리 시작 시간
        start_time = time.time()
        
        # BLIP2로 이미지 캡셔닝
        caption = self.generate_caption(cv_image)
        
        # EfficientSAM으로 세그멘테이션
        segmented_image = self.segment_image(cv_image)
        
        # 처리 시간 계산
        process_time = time.time() - start_time
        
        # 결과 출력
        print("\n" + "="*50)
        print("🎉 처리 완료!")
        print(f"⏱️  처리 시간: {process_time:.2f}초")
        print(f"📝 캡션: {caption}")
        print(f"🎯 세그멘테이션: {'완료' if segmented_image is not None else '실패'}")
        print("="*50)
        
        # 결과 파일 경로 출력
        print(f"📁 결과 파일들:")
        print(f"   - 원본 이미지: {self.image_path}")
        print(f"   - 캡션: {self.caption_path}")
        print(f"   - 세그멘테이션: {self.segmented_path}")

def main():
    """메인 함수"""
    print("🎯 이미지 처리 시스템 시작")
    
    # 프로세서 초기화
    processor = ImageProcessor()
    
    # 이미지 처리 실행
    processor.process_image()
    
    print("\n✅ 모든 처리가 완료되었습니다!")

if __name__ == '__main__':
    main() 