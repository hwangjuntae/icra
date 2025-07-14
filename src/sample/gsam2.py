import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
import sys

# Grounded-SAM-2 경로 추가 및 작업 디렉토리 변경
grounded_sam2_path = '/root/ws/src/risk_nav/src/Grounded-SAM-2'
sys.path.append(grounded_sam2_path)
os.chdir(grounded_sam2_path)  # 작업 디렉토리를 Grounded-SAM-2로 변경

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

"""
Office 환경용 Grounded SAM 2 설정
"""
# 사무실 환경에서 찾을 수 있는 객체들
TEXT_PROMPT = "computer. monitor. keyboard. mouse. chair. desk. lamp. phone. book. paper. pen. cup. bottle."
IMG_PATH = "/root/ws/src/risk_nav/src/sample/office_img.png"

# 모델 경로들 (상대 경로로 변경)
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"

# 임계값 설정
BOX_THRESHOLD = 0.25  # 더 많은 객체를 탐지하기 위해 낮춤
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 출력 디렉토리 설정
OUTPUT_DIR = Path("/root/ws/src/risk_nav/src/sample/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"🏢 Office 이미지 분석을 시작합니다...")
print(f"📁 이미지 경로: {IMG_PATH}")
print(f"🔍 검색할 객체들: {TEXT_PROMPT}")
print(f"💻 디바이스: {DEVICE}")

# SAM2 모델 빌드
print("🤖 SAM2 모델을 로드 중...")
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# Grounding DINO 모델 로드
print("🎯 Grounding DINO 모델을 로드 중...")
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)

# 이미지 로드
print("📷 이미지를 로드 중...")
image_source, image = load_image(IMG_PATH)

# SAM2 이미지 설정
sam2_predictor.set_image(image_source)

# Grounding DINO로 객체 탐지
print("🔍 객체 탐지 중...")
boxes, confidences, labels = predict(
    model=grounding_model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD
)

print(f"✅ {len(boxes)}개의 객체를 탐지했습니다!")

# 탐지된 객체들 출력
for i, (box, conf, label) in enumerate(zip(boxes, confidences, labels)):
    print(f"  {i+1}. {label}: {conf:.3f}")

# 박스를 SAM2 입력 형식으로 변환
h, w, _ = image_source.shape
boxes = boxes * torch.Tensor([w, h, w, h])
input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

# SAM2로 세그멘테이션
print("🖼️  세그멘테이션 수행 중...")
masks, scores, logits = sam2_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)

# 결과 시각화
print("🎨 결과를 시각화 중...")

# 탐지 결과 (박스 + 라벨)
# 마스크 차원을 (N, 1, H, W) -> (N, H, W)로 변경
processed_masks = masks.squeeze(1).astype(bool)

detections = sv.Detections(
    xyxy=input_boxes,
    mask=processed_masks,
    class_id=np.array([i for i in range(len(labels))]),
    confidence=confidences.numpy()  # PyTorch tensor를 numpy 배열로 변환
)

# 박스 어노테이터
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# 라벨 생성
annotated_labels = [
    f"{label} {confidence:.2f}"
    for label, confidence in zip(labels, confidences)
]

# 어노테이션 적용
annotated_image = box_annotator.annotate(
    scene=image_source.copy(), 
    detections=detections
)
annotated_image = label_annotator.annotate(
    scene=annotated_image, 
    detections=detections, 
    labels=annotated_labels
)

# 마스크 오버레이
mask_annotator = sv.MaskAnnotator()
annotated_image_with_mask = mask_annotator.annotate(
    scene=annotated_image, 
    detections=detections
)

# 결과 저장
print("💾 결과를 저장 중...")

# 박스만 있는 이미지
output_path_boxes = OUTPUT_DIR / "office_grounded_dino_result.jpg"
cv2.imwrite(str(output_path_boxes), annotated_image)

# 마스크 포함 이미지
output_path_masks = OUTPUT_DIR / "office_grounded_sam2_result.jpg"
cv2.imwrite(str(output_path_masks), annotated_image_with_mask)

# JSON 결과 저장
results = {
    "image_path": IMG_PATH,
    "text_prompt": TEXT_PROMPT,
    "detections": [],
    "settings": {
        "box_threshold": BOX_THRESHOLD,
        "text_threshold": TEXT_THRESHOLD,
        "device": DEVICE
    }
}

for i, (box, conf, label, mask) in enumerate(zip(input_boxes, confidences, labels, masks)):
    # 마스크를 RLE 형식으로 인코딩 (uint8 타입으로 변환)
    mask_uint8 = (mask[0] * 255).astype(np.uint8)
    mask_rle = mask_util.encode(np.asfortranarray(mask_uint8))
    mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
    
    detection = {
        "id": i + 1,
        "class_name": label,
        "confidence": float(conf),
        "bbox": [float(x) for x in box],
        "bbox_format": "xyxy",
        "mask": {
            "size": mask_rle["size"],
            "counts": mask_rle["counts"]
        }
    }
    results["detections"].append(detection)

# JSON 파일 저장
json_path = OUTPUT_DIR / "office_analysis_results.json"
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("🎉 분석 완료!")
print(f"📊 총 {len(boxes)}개의 객체를 탐지했습니다.")
print(f"📁 결과 파일:")
print(f"  - 박스 결과: {output_path_boxes}")
print(f"  - 마스크 결과: {output_path_masks}")
print(f"  - JSON 결과: {json_path}")

# 탐지된 객체별 통계
object_counts = {}
for label in labels:
    object_counts[label] = object_counts.get(label, 0) + 1

print(f"\n📈 탐지된 객체 통계:")
for obj, count in object_counts.items():
    print(f"  - {obj}: {count}개") 