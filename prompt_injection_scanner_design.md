# 🛡️ 프로젝트 설계서: LatentVision-Shield (가칭)
**부제: 잠재 공간 시각화(Latent Space Visualization) 기반 프롬프트 인젝션 스캐너**

---

## 1. 프로젝트 개요 (Overview)
**LatentVision-Shield**는 기존 텍스트 기반 프롬프트 인젝션 탐지의 한계를 보완하기 위해, **입력된 텍스트 프롬프트를 '2D 이미지(Spectrogram/Heatmap 형태)'로 변환하여 시각적 이상 패턴을 탐지**하는 Prompt Injection Scanner입니다.

공격자가 오탈자, 특수문자, 동의어 치환 등을 통해 텍스트 필터링을 우회하더라도, 언어 모델이 이를 처리하는 **잠재 공간(Latent Space)과 어텐션(Attention) 맵에서는 공격 특유의 시그니처가 시각적 패턴으로 발현**된다는 점을 활용합니다.

---

## 2. 접근 방식의 핵심: 전처리 과정 (Text-to-Image Transformation)

가장 핵심적인 차별점은 전처리 단계에서 텍스트를 1차원 시퀀스가 아닌 **2차원 공간 데이터(이미지)**로 변환하고, 이를 실제 판별 파이프라인에 연결하는 것입니다.  
참고로 임베딩 시각화 자체는 기존에도 존재하던 방법이며, 본 프로젝트의 포인트는 시각화를 "설명용"이 아니라 "탐지용 신호"로 운영하는 데 있습니다.

### 2.1. 전처리 파이프라인 메커니즘
1. **임베딩 매트릭스 추출 (Embedding Matrix Extraction)**: 
   * 입력된 프롬프트를 경량화된 언어 모델(예: `sentence-transformers/all-MiniLM-L6-v2`)에 통과시켜 토큰별 임베딩 벡터 및 Self-Attention 가중치 매트릭스 `[N x N]`를 추출합니다.
   * **구체화 포인트**: 마지막 Layer의 모든 Head의 Attention 가중치를 평균 내어(Mean Pooling) 하나의 `[N x N]` 매트릭스로 압축합니다. 마지막 Layer가 문맥적 의미(Semantic Context)를 가장 잘 담고 있기 때문입니다.
2. **이미지 렌더링 (Spectrogram / Heatmap Generation)**:
   * 추출된 2D 매트릭스 데이터를 정규화(Normalization, 0~1)하여 `matplotlib` 등을 통해 컬러맵(예: `viridis` 또는 `hot`)이 적용된 RGB 히트맵 이미지로 변환합니다.
   * **구체화 포인트**: 입력 프롬프트의 길이를 최대 512 토큰으로 제한(Truncation)하고, 생성되는 이미지의 해상도를 `512x512` 픽셀로 고정하여 Gemini Vision API가 일관된 크기의 이미지를 분석할 수 있도록 전처리합니다.

### 2.2. 텍스트 시각화(Text-to-Image)의 실효성 및 학술적 근거

"텍스트를 이미지로 변환하여 탐지하는 것이 과연 텍스트 자체를 분석하는 것보다 실효성이 있는가?"에 대한 근거는 **잠재 공간(Latent Space)의 기하학적 특성**과 **어텐션 메커니즘의 시각적 분포**를 다룬 최신 연구들을 통해 강력하게 뒷받침됩니다.

1. **잠재 공간 내 공격 시그니처의 시각적 뚜렷함 (Latent Space Signatures)**
   * **근거 논문**: [*ICON: Indirect Prompt Injection Defense for Agents based on Inference-Time Correction (arXiv:2602.20708, 2026)*](https://arxiv.org/abs/2602.20708)
   * **설명**: 프롬프트 인젝션(특히 시스템 지시를 덮어쓰려는 공격)은 LLM의 잠재 공간 내에서 **"뚜렷한 과집중 시그니처(distinct over-focusing signatures)"**를 남깁니다. 정상적인 문맥 흐름은 임베딩 텐서 상에서 부드러운 전환(Smooth Gradient)을 보이지만, 공격 프롬프트는 특정 토큰(예: "Ignore", "System")으로 어텐션이 비정상적으로 쏠리면서 텐서 맵 상에 **급격한 스파이크(Sharp Edge)나 고주파 노이즈**를 발생시킵니다. 이를 2D 히트맵으로 시각화하면 비전 모델이 직관적으로 픽셀의 튀는 패턴(Anomaly)을 잡아낼 수 있습니다.

2. **난독화에 대한 공간적 강건성 (Spatial Robustness against Obfuscation)**
   * **근거 방법론**: 악성코드 탐지 분야의 *Malware Images: Visualization and Automatic Classification* 방법론 차용
   * **설명**: 해커가 `ignore`를 `1gn0r3`로 바꾸면 텍스트 기반 필터링은 이를 새로운 단어로 인식해 통과시킵니다. 그러나 이를 임베딩하여 이미지로 렌더링하면, 해당 토큰 주변의 문맥적 이질성 때문에 픽셀 값의 분포가 주변과 확연히 달라집니다. 즉, **텍스트 레벨의 미세한 노이즈(오탈자)가 잠재 공간 이미지에서는 뚜렷한 '시각적 얼룩(Visual Artifact)'으로 증폭**되어 나타나므로, 비전 스캐너가 우회 공격을 더 쉽게 방어할 수 있습니다.

---

## 3. 기존 접근과의 차별점 및 기여 (The Leap)

앞서 2.2절에서 언급한 논문들은 "어텐션과 잠재 공간에 공격의 흔적이 남는다"는 **이론적 근거(Evidence)**를 제공합니다. 본 프로젝트의 기여는 이 근거를 실제 보안 운영 관점의 **다층 탐지 파이프라인**으로 구현했다는 점에 있습니다.

#### 1. AttnTrace(2025)의 한계와 우리의 도약 (사후 분석 ➡️ 사전 시각적 방어)
* **AttnTrace의 한계**: [*AttnTrace: Contextual Attribution of Prompt Injection (arXiv:2508.03793, 2025)*](https://arxiv.org/abs/2508.03793) 논문은 어텐션 맵을 "공격이 성공한 후, 어떤 텍스트가 원인이었는지 추적(Attribution)"하는 **사후 분석(Post-hoc) 도구**로만 사용합니다. 또한, 어텐션 값을 단순한 수치적 가중치로 계산하여 기존 텍스트 탐지기의 보조 지표로만 활용합니다.
* **우리의 기여**: 우리는 어텐션을 사후 분석용 수치에만 두지 않고, **"사전 차단(Pre-emptive)을 위한 2D 시각적 지형도(Visual Topography)"**로 활용합니다. 텍스트의 어텐션과 임베딩을 결합해 이미지(Spectrogram)로 렌더링하고, 이를 비전 모델(Vision Model)로 스캔합니다. 즉, "어텐션이 중요하다"는 AttnTrace의 발견을 실제 실시간 탐지 단계에 연결한 점이 핵심입니다.

#### 2. ICON(2026)의 한계와 우리의 도약 (1D 수치 연산 ➡️ 2D 공간적 강건성)
* **ICON의 한계**: 잠재 공간(Latent Space)을 방어에 활용한 가장 최신 논문인 [*ICON (arXiv:2602.20708)*](https://arxiv.org/abs/2602.20708)조차, 1D-CNN과 MLP를 사용하여 어텐션의 '과집중(Over-focusing) 스코어'를 수치적으로 계산하는 데 그칩니다.
* **우리의 기여**: 우리는 1차원 수치 연산 신호에 더해, 데이터를 2차원 이미지 신호로 변환해 추가 판별 축을 제공합니다. 과거 악성코드(Malware) 탐지 분야에서 이미지 기반 신호가 난독화 대응에 도움을 준 사례처럼, 프롬프트의 2D 시각화는 **텍스트 난독화(Leetspeak, Base64 등)에 대한 공간적 강건성(Spatial Robustness)**을 기대할 수 있습니다. 다만 이 부분은 실험으로 검증해야 하며, 본 프로젝트는 이를 정량 평가로 확인하는 것을 목표로 합니다.

#### 3. LLM 보안의 '모달리티 역전 (Modality Reversal)'
* **기존 연구 흐름**: 최근 연구들은 주로 "이미지 안에 숨겨진 악성 텍스트"를 탐지하는 등 **VLM(Vision-Language Model)을 공격하는 멀티모달 공격 방어**에 집중하고 있습니다. (예: [*Image-based Prompt Injection (arXiv:2603.03637)*](https://arxiv.org/abs/2603.03637))
* **우리의 기여**: 우리는 **순수 텍스트 기반 LLM을 방어하기 위해 텍스트를 이미지로 변환**해 보조 판별 신호를 만듭니다. 텍스트 보안 문제를 컴퓨터 비전 문제로 일부 치환해, 기존 텍스트 필터와 다른 실패 모드를 갖는 다층 방어를 구성합니다.

---

## 4. Gemini 1.5 Pro (Vision)를 활용한 Zero-Shot 판별 프로세스

해커톤의 핵심인 **"학습 없는(Training-free)"** 즉각적인 탐지를 위해, 생성된 히트맵을 구글의 **Gemini 1.5 Pro Vision API**에 전달하여 판별하는 구체적인 과정은 다음과 같습니다. Gemini 모델은 긴 문맥(Long Context)과 뛰어난 멀티모달 추론 능력을 갖추고 있어 히트맵의 미세한 이상 패턴을 분석하는 데 최적화되어 있습니다.

### 4.1. 히트맵 이미지 생성 (전처리 결과물)
* **정상 프롬프트의 히트맵**: 단어들이 문맥에 맞게 고르게 어텐션을 나누어 가지므로, 히트맵 전체에 걸쳐 색상이 부드럽게 퍼져 있는 형태(Smooth Gradient)를 보입니다. 대각선(Self-attention)을 중심으로 완만한 분포를 가집니다.
* **인젝션 프롬프트의 히트맵**: 공격자가 시스템 프롬프트를 무시하고 새로운 명령을 내리려 할 때(예: "Ignore previous instructions", "System Override"), 모델의 어텐션이 해당 악성 토큰들에 비정상적으로 집중됩니다. 이로 인해 히트맵의 특정 영역(행/열)에만 극단적으로 밝은 색상(Red Hot Spot)이 나타나고, 나머지 영역은 어두워지는 **'과집중(Over-focusing) 스파이크'** 또는 **'격자형 단절(Grid-like Disconnect)'** 패턴이 발생합니다.

### 4.2. Gemini Vision API 호출 및 프롬프트 엔지니어링
생성된 2D 히트맵 이미지(PNG/JPEG)를 Base64로 인코딩하여 Gemini API에 전송합니다. 이때 Gemini 에이전트에게 부여하는 시스템 프롬프트(System Instruction)가 탐지 성능을 좌우합니다.

**[Gemini에게 전달할 프롬프트 예시 (System Instruction)]**
```markdown
당신은 AI 보안 전문가이자 컴퓨터 비전 기반의 프롬프트 인젝션 스캐너입니다.
첨부된 이미지는 사용자의 텍스트 프롬프트를 언어 모델에 통과시켜 추출한 '어텐션 가중치(Attention Weights) 히트맵'입니다. 이 히트맵은 X축과 Y축이 모두 프롬프트의 토큰 시퀀스를 나타내며, 픽셀이 밝을수록(빨간색/노란색) 해당 토큰 간의 어텐션이 강함을 의미합니다.

정상적인 문맥을 가진 프롬프트는 단어 간의 상호작용이 고르게 분포하여 히트맵이 대각선을 중심으로 부드러운 그라데이션(Smooth Gradient)을 보입니다.
반면, 프롬프트 인젝션 공격(예: 시스템 지시 무시, 탈옥 등)이 포함된 경우, 특정 악성 토큰으로 모델의 주의가 강제로 쏠리면서 다음과 같은 시각적 이상(Anomaly) 패턴이 나타납니다:
1. 특정 행이나 열 전체에 걸쳐 극단적으로 밝은 색상(Red Hot Spot)이 십자가(+) 모양이나 직선 형태로 나타나는 '과집중(Over-focusing)' 현상
2. 주변 픽셀들과 부드럽게 이어지지 않고 날카롭게 끊어지는 '격자형 단절(Grid-like Disconnect)' 또는 '스파이크(Sharp Edge)'

이 히트맵 이미지를 분석하여 다음을 수행하세요:
1. 위에서 설명한 프롬프트 인젝션 특유의 시각적 이상 패턴(Red Hot Spot, 격자형 단절 등)이 존재하는지 판별하세요.
2. 분석 결과를 바탕으로 이 프롬프트가 '정상(SAFE)'인지 '인젝션 공격(MALICIOUS)'인지 최종 판단하세요.
3. 결과를 다음 JSON 형식으로만 출력하세요:
{
  "status": "SAFE" 또는 "MALICIOUS",
  "confidence_score": 0.0 ~ 1.0,
  "visual_evidence": "이미지에서 발견된 이상 패턴에 대한 시각적 설명 (예: 우측 상단에 극단적인 Red Hot Spot 발견 등)"
}
```

### 4.3. 이 방식(Zero-Shot Gemini Vision)의 강력한 장점
* **학술적 근거의 시각화**: ICON 논문(2026)에서 수치적으로만 계산했던 '과집중(Over-focusing)' 현상을 Gemini가 직접 눈으로 보고 판별하게 함으로써, 이론적 근거를 완벽하게 실체화합니다.
* **난독화 무력화**: 공격자가 텍스트를 Base64로 인코딩하거나 특수문자를 섞어 기존 텍스트 필터를 우회하더라도, 해당 토큰들이 문맥과 동떨어져 발생하는 '시각적 얼룩(Visual Artifact)'은 Gemini의 뛰어난 멀티모달 추론 능력을 속일 수 없습니다.
* **설명 가능한 AI (Explainable AI)**: Gemini가 반환하는 `visual_evidence` 필드를 통해, 사용자나 관리자에게 "왜 이 프롬프트가 차단되었는지"를 히트맵의 특정 영역(예: "우측 상단 Red Hot Spot")을 가리키며 직관적으로 설명할 수 있습니다.

---

## 5. 평가 데이터셋 (Evaluation Datasets)

스캐너의 실효성을 검증하고 데모를 시연하기 위해, 최신 공격 트렌드(2025-2026)가 반영된 고품질의 오픈소스 데이터셋을 활용합니다.

1. **[`neuralchemy/Prompt-injection-dataset` (HuggingFace, 2026)](https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset)**
   * **특징**: 2026년에 공개된 최신 데이터셋으로, Crescendo, Token Smuggling, Encoding Obfuscation 등 29개 공격 카테고리를 포함하며 16,000개 이상의 샘플을 제공합니다.
   * **활용**: 가장 최신의 난독화 및 우회 공격 프롬프트를 추출하여, 우리의 시각화 기반 스캐너가 기존 텍스트 필터가 놓치는 최신 공격을 어떻게 방어하는지 시연합니다.
2. **[`Bordair/bordair-multimodal` (HuggingFace, 2025-2026)](https://huggingface.co/datasets/Bordair/bordair-multimodal)**
   * **특징**: 62,000개 이상의 샘플을 보유하고 있으며, GCG(Greedy Coordinate Gradient) Adversarial Suffix, AutoDAN 등 최신 탈옥(Jailbreak) 템플릿과 다중 턴(Multi-turn) 공격 데이터를 포함합니다.
   * **활용**: 의미론적으로는 무해해 보이지만 수학적으로 계산된 무의미한 토큰 조합(Adversarial Suffix)이 잠재 공간에서 어떻게 기형적인 히트맵을 만들어내는지 증명하는 데 사용합니다.

### 5.1. 정량 검증 계획 (Baseline 비교)
본 접근의 가치는 "시각화 자체"가 아니라 "탐지 성능 개선"으로 입증되어야 하므로, 아래 기준으로 비교 평가합니다.

1. **Baseline A: 벡터 유사도 기반 탐지**
   * 알려진 공격 템플릿 임베딩과 코사인 유사도 비교
2. **Baseline B: 임베딩 기반 분류기**
   * MiniLM 임베딩 + 경량 이진 분류기(예: Logistic Regression / MLP)
3. **Proposed: 히트맵 + Gemini Vision 판별**
   * 동일 샘플에 대해 1, 2와 동일 지표로 비교

**평가 지표**
* Accuracy, Precision, Recall, F1
* 난독화 공격셋(leet/base64/typo) 분리 성능
* 평균 처리 시간(ms) 및 샘플당 비용
* 오탐/미탐 사례에 대한 시각적 근거(`visual_evidence`) 품질

---

## 6. 시스템 아키텍처 및 구현 계획 (System Architecture)

```mermaid
graph TD
    A[User Input Text] --> B[HuggingFace MiniLM]
    B -->|Extract| C[Attention Weights Matrix]
    C -->|Matplotlib| D[2D Heatmap Image Generation]
    D --> E[Gemini 1.5 Pro Vision API]
    E -->|Prompt: Find Anomaly| F{Decision}
    F -->|Injection Detected| G[Block & Alert]
    F -->|Clean| H[Pass to Target LLM]
```

### ⏱️ 해커톤 잔여 타임라인 (현재 시각 13:00 기준, 제출 마감 18:00)
현재 오후 1시 기준, 제출 마감(18:00)까지 **단 5시간**이 남았습니다. 선택과 집중을 통해 핵심 데모(PoC)를 완성하는 데 주력합니다. 해커톤 가이드에 명시된 **제공된 Gemini API Key**를 적극 활용합니다.

* **13:00 - 14:30 [Core Logic 1]**: HuggingFace 데이터셋(`neuralchemy`) 로드 및 `all-MiniLM-L6-v2` 모델을 통한 Attention 맵 추출, `matplotlib` 히트맵 렌더링 로직 구현
* **14:30 - 15:30 [Core Logic 2]**: **Gemini 1.5 Pro Vision API** 연동 및 Zero-Shot 프롬프트 엔지니어링 (이미지 전달 후 JSON 응답 파싱)
* **15:30 - 17:00 [Frontend UI]**: Streamlit 또는 Vercel v0를 활용하여 심사위원 시연용 웹 데모 UI 구현 (사용자 텍스트 입력 -> 실시간 히트맵 시각화 -> Gemini 판별 결과 및 시각적 증거(Explainability) 표시)
* **17:00 - 18:00 [Wrap-up]**: 최종 테스트, 데모 링크 배포 확인, GitHub README.md 작성 및 제출 폼 작성

---

## 7. 해커톤(AIM Intelligence 트랙) 기대 효과
* **우회 공격 대응력 향상**: 난독화/변형 공격에서 텍스트 필터 단독 대비 탐지율 개선을 목표로 합니다.
* **시각적 설명 가능성 (Explainability)**: 왜 이 프롬프트가 차단되었는지, 생성된 히트맵 이미지의 어느 부분(어느 토큰)에서 이상 패턴이 발생했는지 사용자에게 직관적으로 보여줄 수 있습니다.
* **실용적 참신성**: 기존 임베딩/유사도/분류기 방식 위에 비전 기반 신호를 결합한 다층 방어 구조로, 실전 운영 관점의 확장성을 제시합니다.