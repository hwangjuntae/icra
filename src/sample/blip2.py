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
    
    # GPU 메모리 최적화 설정
    if torch.cuda.is_available():
        # 강력한 GPU 메모리 정리
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # PyTorch 메모리 할당 최적화
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        # 메모리 효율적인 할당 설정
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        device = torch.device("cuda")
        print(f"사용 디바이스: {device}")
        print("GPU 메모리 최적화 완료")
    else:
        device = torch.device("cpu")
        print(f"사용 디바이스: {device}")

    try:
        # 이미지 로드
        print("이미지 로딩 중...")
        raw_image = Image.open("office_img.png").convert("RGB")
        print(f"이미지 로드 완료: {raw_image.size}")

        # VQA 모델 로드 (Visual Question Answering)
        print("VQA 모델 로딩 중...")
        
        # 메모리 정리 후 모델 로딩
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # VQA를 위한 BLIP 모델 로드
        model, vis_processors, txt_processors = load_model_and_preprocess(
            name="blip_vqa",
            model_type="vqav2",
            is_eval=True,
            device=device
        )
        
        # 다른 VQA 모델 옵션들:
        # BLIP-2 VQA 모델
        # model, vis_processors, txt_processors = load_model_and_preprocess(
        #     name="blip2_t5",
        #     model_type="pretrain_flant5xl",
        #     is_eval=True,
        #     device=device
        # )
        
        model.eval()
        
        # 메모리 최적화: 그래디언트 비활성화
        for param in model.parameters():
            param.requires_grad = False

        # 최적화된 추론
        with torch.no_grad():
            print("모델 로드 완료!")
            
            # 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 이미지 전처리
            image = vis_processors["eval"](raw_image).unsqueeze(0)
            
            # GPU 사용 시 이미지를 GPU로 이동
            image = image.to(device)

            # ============ 여기서 질문을 변경할 수 있습니다 ============
            question = "what is in the picture?"
            
            # 다른 질문 예시들:
            # question = "How many people are in this office?"
            # question = "What is the most important object in this image?"
            # question = "What kind of place is this?"
            # question = "What are the main colors in this image?"
            # question = "Are there any dangerous elements in this photo?"
            
            print(f"질문: {question}")
            # ================================================
            
            # 질문 전처리
            processed_question = txt_processors["eval"](question)

            print("이미지 분석 중...")
            
            # 메모리 효율적인 추론
            try:
                # VQA 모델의 predict_answers 메서드 시도
                samples = {"image": image, "text_input": processed_question}
                result = model.predict_answers(
                    samples,
                    num_beams=3,
                    inference_method="generate",
                    max_len=30,
                    min_len=1,
                    num_ans_candidates=128,
                    answer_list=None,
                )
            except Exception as e:
                print(f"predict_answers 메서드 오류: {e}")
                # 다른 방법으로 시도
                try:
                    # 간단한 forward pass 시도
                    samples = {"image": image, "text_input": processed_question}
                    result = model(samples)
                    if hasattr(result, 'prediction'):
                        result = result.prediction
                    elif hasattr(result, 'predictions'):
                        result = result.predictions
                    else:
                        result = [str(result)]
                except Exception as e2:
                    print(f"모델 호출 오류: {e2}")
                    result = ["모델 호출 실패"]
            
            # 결과 처리 및 출력
            if isinstance(result, list):
                if len(result) > 0:
                    answer = result[0]
                    print(f"답변: {answer}")
                else:
                    print("답변: 빈 결과가 반환되었습니다.")
            else:
                print(f"답변: {result}")

    except Exception as e:
        print(f"오류 발생: {e}")

    finally:
        # 강력한 메모리 정리
        try:
            if 'model' in locals():
                del model
            if 'vis_processors' in locals():
                del vis_processors
            if 'txt_processors' in locals():
                del txt_processors
            if 'image' in locals():
                del image
            if 'raw_image' in locals():
                del raw_image
            if 'result' in locals():
                del result
        except:
            pass
            
        # GPU 메모리 완전 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
        
        gc.collect()

if __name__ == "__main__":
    main()