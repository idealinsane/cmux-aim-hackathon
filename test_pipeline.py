import os
import json
from core.preprocessor import PromptPreprocessor
from core.scanner import LatentVisionScanner

def test_pipeline():
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
        
    samples = samples[:2] + samples[50:52]

    for i, test in enumerate(samples, 1):
        print(f"\n--- 테스트 케이스 {i} ---")
        print(f"입력: {test['text'][:50]}")
        
        try:
            print("  -> 히트맵 생성 중...")
            heatmap = preprocessor.process(test['text'])
            print(f"  -> 히트맵 생성 완료 (크기: {heatmap.size})")
            
            print("  -> Gemini 스캔 중...")
            result = scanner.scan(heatmap)
            print(f"  -> 스캔 결과: {result}")
        except Exception as e:
            print(f"  -> ❌ 테스트 실패: {e}")

if __name__ == "__main__":
    test_pipeline()
