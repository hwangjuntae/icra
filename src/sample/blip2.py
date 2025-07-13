import torch
import gc
import os
from PIL import Image
from lavis.models import load_model_and_preprocess

def main():
    print("=== BLIP-2 이미지 분석 시작 ===")

    # 환경 최적화
    torch.set_num_threads(4)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = torch.device("cpu")
    print(f"사용 디바이스: {device}")

    try:
        # 이미지 로드
        print("이미지 로딩 중...")
        raw_image = Image.open("dog.jpg").convert("RGB")
        print(f"이미지 로드 완료: {raw_image.size}")

        # 모델 로드
        print("모델 로딩 중...")
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_opt",
            model_type="caption_coco_opt2.7b",
            is_eval=True,
            device=device
        )
        model.eval()

        # 최적화된 추론
        with torch.no_grad():
            print("모델 로드 완료!")
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

            print("이미지 분석 중...")
            result = model.generate({
                "image": image,
                "prompt": "what is in the picture?",
                "max_length": 20,
                "num_beams": 1,
                "do_sample": False
            })
            print(f"결과: {result[0] if isinstance(result, list) else result}")

    except Exception as e:
        print(f"오류 발생: {e}")

    finally:
        # 메모리 정리
        if 'model' in locals():
            del model
        if 'image' in locals():
            del image
        if 'raw_image' in locals():
            del raw_image
        gc.collect()
        print("완료!")

if __name__ == "__main__":
    main()