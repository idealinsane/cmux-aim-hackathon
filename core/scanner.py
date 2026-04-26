import os
import json
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

class LatentVisionScanner:
    def __init__(self):
        # GEMINI_API_KEY_4를 우선 사용 (1~3번 키 유출 문제 회피)
        api_key = os.getenv("GEMINI_API_KEY_4") or os.getenv("GEMINI_API_KEY_5")
        if not api_key:
            raise ValueError("Gemini API Key가 .env 파일에 설정되지 않았습니다.")
        
        genai.configure(api_key=api_key)
        
        # 시스템 프롬프트 설정 (설계서 기준)
        self.system_instruction = """당신은 AI 보안 전문가이자 컴퓨터 비전 기반의 프롬프트 인젝션 스캐너입니다.
첨부된 이미지는 사용자의 텍스트 프롬프트를 언어 모델에 통과시켜 추출한 '어텐션 가중치(Attention Weights) 히트맵'입니다. 이 히트맵은 X축과 Y축이 모두 프롬프트의 토큰 시퀀스를 나타내며, 픽셀이 밝을수록(빨간색/노란색) 해당 토큰 간의 어텐션이 강함을 의미합니다.

정상적인 문맥을 가진 프롬프트는 단어 간의 상호작용이 고르게 분포하여 히트맵이 대각선을 중심으로 부드러운 그라데이션(Smooth Gradient)을 보입니다.
반면, 프롬프트 인젝션 공격(예: 시스템 지시 무시, 탈옥 등)이 포함된 경우, 특정 악성 토큰으로 모델의 주의가 강제로 쏠리면서 다음과 같은 시각적 이상(Anomaly) 패턴이 나타납니다:
1. 특정 행이나 열 전체에 걸쳐 극단적으로 밝은 색상(Red Hot Spot)이 십자가(+) 모양이나 직선 형태로 나타나는 '과집중(Over-focusing)' 현상
2. 주변 픽셀들과 부드럽게 이어지지 않고 날카롭게 끊어지는 '격자형 단절(Grid-like Disconnect)' 또는 '스파이크(Sharp Edge)'

주의사항 (False Positive 방지):
- 입력된 문장의 길이가 짧을 경우, 히트맵 전체가 몇 개의 큰 블록으로 나뉘어 보이거나, 핵심 키워드(명사 등)에 어텐션이 집중되어 세로 방향의 밝은 띠(Vertical Stripe)가 나타날 수 있습니다.
- 이는 자연어 처리의 정상적인 특성이므로, 문맥과 무관하게 극단적으로 튀는 바늘 같은 스파이크(Sharp Spike)나 완벽한 십자가(+) 모양이 아니라면 'SAFE'로 판별하세요.
- 전체적으로 부드럽게 이어지는 띠 모양은 악성 공격이 아닙니다.

이 히트맵 이미지를 분석하여 다음을 수행하세요:
1. 위에서 설명한 프롬프트 인젝션 특유의 시각적 이상 패턴(Red Hot Spot, 격자형 단절 등)이 존재하는지 판별하세요.
2. 분석 결과를 바탕으로 이 프롬프트가 '정상(SAFE)'인지 '인젝션 공격(MALICIOUS)'인지 최종 판단하세요.
3. 결과를 다음 JSON 형식으로만 출력하세요:
{
  "status": "SAFE" 또는 "MALICIOUS",
  "confidence_score": 0.0 ~ 1.0,
  "visual_evidence": "이미지에서 발견된 이상 패턴에 대한 시각적 설명 (예: 우측 상단에 극단적인 Red Hot Spot 발견 등)"
}"""

        # JSON 스키마 정의 (이진 출력 강제)
        response_schema = {
            "type": "OBJECT",
            "properties": {
                "status": {
                    "type": "STRING",
                    "enum": ["SAFE", "MALICIOUS"], # 두 가지 값 중 하나만 출력하도록 강제 (이진 분류)
                    "description": "프롬프트 인젝션 여부"
                },
                "is_injection": {
                    "type": "BOOLEAN", # True/False 형태의 명시적 이진 출력
                    "description": "인젝션 공격이면 true, 정상이면 false"
                },
                "confidence_score": {
                    "type": "NUMBER"
                },
                "visual_evidence": {
                    "type": "STRING"
                }
            },
            "required": ["status", "is_injection", "confidence_score", "visual_evidence"]
        }

        # gemini-3-flash-preview 모델 초기화
        # JSON 응답을 강제하기 위해 response_mime_type="application/json" 및 response_schema 설정
        self.model = genai.GenerativeModel(
            model_name="gemini-3-flash-preview",
            system_instruction=self.system_instruction,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                response_schema=response_schema
            )
        )

    def scan(self, heatmap_image: Image.Image, quantitative_confidence: float = None) -> dict:
        """
        히트맵 이미지를 Gemini API에 전송하여 인젝션 여부를 스캔합니다.
        
        Args:
            heatmap_image (PIL.Image.Image): 전처리된 2D 히트맵 이미지
            quantitative_confidence (float, optional): 전처리기에서 계산된 정량적 신뢰도 점수 (0.0~1.0)
            
        Returns:
            dict: 스캔 결과 (status, confidence_score, visual_evidence)
        """
        try:
            # Gemini 모델 호출
            response = self.model.generate_content([heatmap_image])
            
            # JSON 응답 텍스트 파싱
            result_text = response.text
            result = json.loads(result_text)
            
            # Gemini가 반환한 신뢰도를 무시하고, 전처리기에서 계산된 정량적 신뢰도로 덮어쓰기
            # 만약 quantitative_confidence가 제공되지 않았다면 기존처럼 Gemini의 값을 사용
            if quantitative_confidence is not None:
                result["confidence_score"] = quantitative_confidence
            
            return result
        except json.JSONDecodeError as e:
            # JSON 파싱 실패 시 기본 오류 결과 반환
            return {
                "status": "ERROR",
                "confidence_score": quantitative_confidence if quantitative_confidence is not None else 0.0,
                "visual_evidence": f"결과 파싱 중 오류 발생: {str(e)}\n응답: {response.text}"
            }
        except Exception as e:
            # 기타 API 호출 오류 처리
            return {
                "status": "ERROR",
                "confidence_score": quantitative_confidence if quantitative_confidence is not None else 0.0,
                "visual_evidence": f"스캔 중 오류 발생: {str(e)}"
            }
