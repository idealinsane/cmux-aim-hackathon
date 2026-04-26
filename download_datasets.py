import os
import json
from datasets import load_dataset

def download_and_save_datasets():
    os.makedirs("data", exist_ok=True)
    
    print("1. neuralchemy/Prompt-injection-dataset 다운로드 중...")
    try:
        # 이 데이터셋은 주로 텍스트 프롬프트와 인젝션 여부 라벨을 포함합니다.
        dataset1 = load_dataset("neuralchemy/Prompt-injection-dataset")
        
        # train split에서 일부 샘플 추출 (예: 100개)
        # 전체를 저장하기엔 클 수 있으므로 테스트용으로 추출
        samples1 = []
        for i, item in enumerate(dataset1['train']):
            if i >= 100:
                break
            samples1.append(item)
            
        with open("data/neuralchemy_samples.json", "w", encoding="utf-8") as f:
            json.dump(samples1, f, ensure_ascii=False, indent=2)
        print(f"✅ neuralchemy 데이터셋 100개 샘플 저장 완료 (data/neuralchemy_samples.json)")
        
    except Exception as e:
        print(f"❌ neuralchemy 데이터셋 다운로드 실패: {e}")

    print("\n2. Bordair/bordair-multimodal 다운로드 중...")
    try:
        # 이 데이터셋은 멀티모달 탈옥 템플릿 등을 포함합니다.
        # 데이터셋의 구조가 일관되지 않아(column mismatch) pandas를 통해 부분적으로 로드하거나
        # 특정 split/subset만 로드하도록 시도합니다.
        # 먼저 default 설정으로 로드 시도
        dataset2 = load_dataset("Bordair/bordair-multimodal", split="train", ignore_verifications=True, trust_remote_code=True)
        
        samples2 = []
        for i, item in enumerate(dataset2):
            if i >= 100:
                break
            # 텍스트 필드만 추출 시도
            serializable_item = {}
            for k, v in item.items():
                if isinstance(v, (str, int, float, bool, list, dict)):
                    serializable_item[k] = v
            samples2.append(serializable_item)
            
        with open("data/bordair_samples.json", "w", encoding="utf-8") as f:
            json.dump(samples2, f, ensure_ascii=False, indent=2)
        print(f"✅ Bordair 데이터셋 100개 샘플 저장 완료 (data/bordair_samples.json)")
        
    except Exception as e:
        print(f"❌ Bordair 데이터셋 다운로드 실패: {e}")
        print("  -> 데이터셋 구조 불일치로 인해 다운로드에 실패했습니다. (Column mismatch)")
        print("  -> 대안: 다른 데이터셋(예: jailbreak_prompts)을 다운로드합니다.")
        try:
            # 대안 데이터셋 다운로드
            alt_dataset = load_dataset("rubend18/ChatGPT-Jailbreak-Prompts")
            samples2 = []
            for i, item in enumerate(alt_dataset['train']):
                if i >= 100:
                    break
                samples2.append(item)
                
            with open("data/jailbreak_samples.json", "w", encoding="utf-8") as f:
                json.dump(samples2, f, ensure_ascii=False, indent=2)
            print(f"✅ 대안 데이터셋(rubend18/ChatGPT-Jailbreak-Prompts) 100개 샘플 저장 완료 (data/jailbreak_samples.json)")
        except Exception as e2:
            print(f"❌ 대안 데이터셋 다운로드도 실패했습니다: {e2}")

if __name__ == "__main__":
    download_and_save_datasets()
