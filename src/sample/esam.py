import sys
import os

# EfficientSAM 폴더를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
efficient_sam_path = os.path.join(current_dir, "EfficientSAM")
sys.path.insert(0, efficient_sam_path)

# 작업 디렉토리를 EfficientSAM으로 변경
original_cwd = os.getcwd()
os.chdir(efficient_sam_path)

from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import matplotlib.pyplot as plt

# 모델 로드
model = build_efficient_sam_vitt()  # 경량 버전
# 또는
# model = build_efficient_sam_vits()  # 표준 버전

# 작업 디렉토리를 원래대로 복원
os.chdir(original_cwd)

# 이미지 로드 (현재 디렉토리의 dog.jpg)
sample_image_np = np.array(Image.open("dog.jpg"))
print(f"이미지 크기: {sample_image_np.shape}")
sample_image_tensor = transforms.ToTensor()(sample_image_np)

# 이미지 크기에 맞는 포인트 입력 (개의 중앙 부분을 클릭)
height, width = sample_image_np.shape[:2]
input_points = torch.tensor([[[[width//2, height//2]]]])  # 이미지 중앙점
input_labels = torch.tensor([[[1]]])  # 전경 포인트

print(f"입력 포인트: {input_points}")

# 추론 실행
predicted_logits, predicted_iou = model(
    sample_image_tensor[None, ...],
    input_points,
    input_labels,
)

# 결과 처리 및 시각화
sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
predicted_logits = torch.take_along_dim(
    predicted_logits, sorted_ids[..., None, None], dim=2
)

# 마스크 생성
mask = torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy()
print(f"마스크 형태: {mask.shape}")
print(f"마스크에서 True인 픽셀 수: {np.sum(mask)} / {mask.size}")

# 1. 순수 마스크 저장 (흰색=전경, 검은색=배경)
mask_image = (mask * 255).astype(np.uint8)
Image.fromarray(mask_image).save("pure_mask.png")

# 2. 마스크가 적용된 이미지 생성 (기존 방식)
masked_image_np = sample_image_np.copy().astype(np.uint8) * mask[:,:,None]
Image.fromarray(masked_image_np).save("result_mask.png")

# 3. 원본 이미지에 마스크 오버레이 (빨간색으로 표시)
overlay_image = sample_image_np.copy().astype(np.uint8)
overlay_image[mask] = [255, 0, 0]  # 마스크 영역을 빨간색으로
Image.fromarray(overlay_image).save("overlay_mask.png")

# 4. 반투명 오버레이
alpha = 0.5
overlay_alpha = sample_image_np.copy().astype(np.float32)
overlay_alpha[mask] = overlay_alpha[mask] * (1 - alpha) + np.array([255, 0, 0]) * alpha
overlay_alpha = overlay_alpha.astype(np.uint8)
Image.fromarray(overlay_alpha).save("alpha_overlay_mask.png")

# 결과 출력
print(f"결과 파일들이 저장되었습니다:")
print(f"- pure_mask.png: 순수 마스크 (흰색=전경)")
print(f"- result_mask.png: 마스크 적용된 이미지")
print(f"- overlay_mask.png: 빨간색 오버레이")
print(f"- alpha_overlay_mask.png: 반투명 빨간색 오버레이")
print(f"IoU 점수: {predicted_iou[0, 0, 0].item():.4f}")