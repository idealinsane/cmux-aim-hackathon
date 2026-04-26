import os
import json
from datasets import load_dataset

def extract_balanced_dataset():
    os.makedirs("data", exist_ok=True)
    
    print("1. neuralchemy/Prompt-injection-dataset 로드 중...")
    try:
        dataset = load_dataset("neuralchemy/Prompt-injection-dataset")
        
        benign_samples = []
        malicious_samples = []
        
        for item in dataset['train']:
            if item['label'] == 0 and len(benign_samples) < 50:
                benign_samples.append(item)
            elif item['label'] == 1 and len(malicious_samples) < 50:
                malicious_samples.append(item)
                
            if len(benign_samples) == 50 and len(malicious_samples) == 50:
                break
                
        balanced_samples = benign_samples + malicious_samples
        
        with open("data/neuralchemy_balanced_100.json", "w", encoding="utf-8") as f:
            json.dump(balanced_samples, f, ensure_ascii=False, indent=2)
            
        print(f"✅ 정상 50개, 악성 50개 총 100개 샘플 추출 완료 (data/neuralchemy_balanced_100.json)")
        
    except Exception as e:
        print(f"❌ 데이터셋 추출 실패: {e}")

if __name__ == "__main__":
    extract_balanced_dataset()
