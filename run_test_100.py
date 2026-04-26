import json
import time
import os
from core.preprocessor import PromptPreprocessor
from core.scanner import LatentVisionScanner

def main():
    dataset_path = "data/neuralchemy_balanced_100.json"
    if not os.path.exists(dataset_path):
        print(f"❌ 데이터셋 파일을 찾을 수 없습니다: {dataset_path}")
        return
        
    print("1. 데이터셋 로드 중...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
        
    print("\n2. 스캐너 시스템 초기화 중...")
    try:
        preprocessor = PromptPreprocessor()
        scanner = LatentVisionScanner()
        print("✅ 초기화 성공")
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        return

    print(f"\n3. 총 {len(samples)}개 샘플 테스트 시작...")
    
    results = {
        "true_positive": 0,
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0,
        "errors": 0
    }
    
    detailed_results = []
    
    for i, sample in enumerate(samples, 1):
        text = sample['text']
        actual_label = "MALICIOUS" if sample['label'] == 1 else "SAFE"
        
        print(f"\n[{i}/100] 실제: {actual_label} | 텍스트: {text[:50]}...")
        
        try:
            # 전처리기에서 히트맵과 정량적 신뢰도를 모두 반환받음
            heatmap, quant_confidence = preprocessor.process(text)
            
            # 스캐너에 히트맵과 정량적 신뢰도를 함께 전달
            # API Rate Limit 방지를 위해 Gemini 호출 전에만 대기
            if i % 15 == 0:
                print("  (Rate limit 방지를 위해 5초 대기...)")
                time.sleep(5)
            else:
                time.sleep(1)
                
            scan_result = scanner.scan(heatmap, quantitative_confidence=quant_confidence)
            predicted_label = scan_result.get("status", "ERROR")
            
            print(f"  -> 예측: {predicted_label} (신뢰도: {scan_result.get('confidence_score', 0.0)})")
            
            if predicted_label == "ERROR":
                results["errors"] += 1
                print(f"  -> ⚠️ 에러: {scan_result.get('visual_evidence')}")
            elif actual_label == "MALICIOUS" and predicted_label == "MALICIOUS":
                results["true_positive"] += 1
            elif actual_label == "SAFE" and predicted_label == "SAFE":
                results["true_negative"] += 1
            elif actual_label == "SAFE" and predicted_label == "MALICIOUS":
                results["false_positive"] += 1
                print("  -> ❌ 오탐 (False Positive)")
            elif actual_label == "MALICIOUS" and predicted_label == "SAFE":
                results["false_negative"] += 1
                print("  -> ❌ 미탐 (False Negative)")
                
            detailed_results.append({
                "id": i,
                "text": text,
                "actual": actual_label,
                "predicted": predicted_label,
                "confidence": scan_result.get("confidence_score", 0.0),
                "evidence": scan_result.get("visual_evidence", "")
            })
            
        except Exception as e:
            print(f"  -> ❌ 테스트 실패: {e}")
            results["errors"] += 1
            
    print("\n" + "="*40)
    print("📊 최종 테스트 결과 요약")
    print("="*40)
    print(f"총 테스트 수: {len(samples)}")
    print(f"✅ True Positive (악성 탐지 성공): {results['true_positive']}")
    print(f"✅ True Negative (정상 통과 성공): {results['true_negative']}")
    print(f"❌ False Positive (정상을 악성으로 오탐): {results['false_positive']}")
    print(f"❌ False Negative (악성을 정상으로 미탐): {results['false_negative']}")
    print(f"⚠️ 에러 발생: {results['errors']}")
    
    valid_tests = len(samples) - results['errors']
    if valid_tests > 0:
        accuracy = (results['true_positive'] + results['true_negative']) / valid_tests * 100
        print(f"\n🎯 정확도(Accuracy): {accuracy:.2f}%")
        
    with open("data/test_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "summary": results,
            "details": detailed_results
        }, f, ensure_ascii=False, indent=2)
    print("\n상세 결과가 data/test_results.json에 저장되었습니다.")

if __name__ == "__main__":
    main()
