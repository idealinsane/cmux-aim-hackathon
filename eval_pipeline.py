import os
import json
import time
from core.preprocessor import PromptPreprocessor
from core.scanner import LatentVisionScanner

def main():
    print("1. 초기화 중...")
    try:
        preprocessor = PromptPreprocessor()
        scanner = LatentVisionScanner()
        print("✅ 초기화 성공")
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        return

    dataset_path = "data/neuralchemy_balanced_100.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
        
    samples = samples[:5] + samples[50:55]
    print(f"\n2. 총 {len(samples)}개 샘플 테스트 시작...")
    
    for i, sample in enumerate(samples, 1):
        text = sample['text']
        actual = "MALICIOUS" if sample['label'] == 1 else "SAFE"
        print(f"\n[{i}/10] 실제: {actual} | 텍스트: {text[:50]}...")
        
        try:
            time.sleep(2)
            heatmap = preprocessor.process(text)
            res = scanner.scan(heatmap)
            pred = res.get("status", "ERROR")
            print(f"  -> 예측: {pred} (신뢰도: {res.get('confidence_score', 0.0)})")
        except Exception as e:
            print(f"  -> ❌ 테스트 실패: {e}")

if __name__ == "__main__":
    main()
