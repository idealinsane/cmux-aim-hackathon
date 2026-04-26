import io
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from PIL import Image

class PromptPreprocessor:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # output_attentions=True를 설정하여 Attention 가중치를 추출할 수 있도록 모델 로드
        self.model = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.model.eval()
        
        # CPU/GPU 설정 (MPS 버그 방지를 위해 일단 CPU로 고정)
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def process(self, text: str) -> tuple[Image.Image, float]:
        """
        입력 텍스트를 처리하여 Attention 히트맵 이미지와 정량적 신뢰도 점수를 반환합니다.
        
        Returns:
            tuple: (히트맵 이미지, 정량적 신뢰도 점수 0.0~1.0)
        """
        # 1. 텍스트 토큰화 (최대 512 토큰 제한)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # 2. 모델 추론 및 Attention 추출
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # outputs.attentions는 (layer_count) 길이의 튜플, 각 요소는 (batch, num_heads, seq_len, seq_len)
        # sentence-transformers/all-MiniLM-L6-v2 모델의 중간 레이어(예: 3번째 레이어) Attention 추출
        target_layer_attention = outputs.attentions[3] # shape: (1, num_heads, seq_len, seq_len)
        
        # 3. 모든 Head의 Attention 가중치를 최댓값으로 풀링 (Max Pooling)
        # 특정 Head에서 발생한 강한 인젝션 스파이크 신호가 평균(Mean)에 의해 희석되는 것을 방지
        # shape: (seq_len, seq_len)
        # CPU로 옮겨서 연산하여 메모리 부족 방지
        pooled_attention = target_layer_attention.squeeze(0).max(dim=0)[0].cpu()
        
        # 3.5. 특수 토큰([CLS], [SEP]) 제외 (Slicing)
        # 문장 시작([CLS])과 끝([SEP])에 과도하게 집중되는 Attention 스파이크를 제거하여 오탐(False Positive) 방지
        # seq_len이 3 이상일 때만 양 끝 토큰 제외
        if pooled_attention.shape[0] > 2:
            pooled_attention = pooled_attention[1:-1, 1:-1]
            
        # --- 정량적 신뢰도(Confidence Score) 계산 로직 추가 ---
        # 프롬프트 인젝션의 특징인 '과집중(Over-focusing)'을 수학적으로 측정
        # 특정 행(Row)이나 열(Column)에 어텐션이 비정상적으로 쏠리는 현상을 분산(Variance)과 최대값(Max)으로 평가
        
        # 1. 각 토큰이 다른 토큰들로부터 받는 어텐션의 합 (열 방향 합)
        col_sums = pooled_attention.sum(dim=0)
        # 2. 어텐션 집중도의 불균형(분산) 측정 (스칼라 값일 경우 0으로 처리)
        if col_sums.numel() > 1:
            # torch.var는 float 텐서에서만 작동하므로 float 변환
            # unbiased=False를 설정하여 1D 텐서에서도 안정적으로 분산 계산
            # CPU로 옮겨서 계산하여 혹시 모를 디바이스 동기화 이슈 방지
            attention_variance = torch.var(col_sums.float().cpu(), unbiased=False).item()
        else:
            attention_variance = 0.0
        # 3. 가장 강한 어텐션 스파이크의 크기
        max_attention_spike = torch.max(pooled_attention).item()
        
        # 분산과 최대 스파이크 값을 조합하여 0.0 ~ 1.0 사이의 점수로 정규화 (경험적 임계값 적용)
        # 분산이 클수록(특정 단어에 쏠림), 최대 스파이크가 클수록 인젝션 확률이 높다고 판단
        # (임계값은 데이터셋 특성에 따라 튜닝 필요)
        # 텐서의 평균값이 1.0에 가깝기 때문에 분산 임계값을 0.5로, 스파이크 임계값을 1.0으로 조정
        norm_variance = min(attention_variance / 0.5, 1.0)  
        norm_spike = min(max_attention_spike / 1.0, 1.0)    
        
        # 두 지표의 가중 평균으로 최종 정량적 신뢰도 산출
        quantitative_confidence = (norm_variance * 0.6) + (norm_spike * 0.4)
        # ---------------------------------------------------
            
        # 3.8. 텐서 리사이징 (Tensor Interpolation) - 길이 불변성(Length Invariance) 확보
        # 프롬프트 길이에 상관없이 항상 256x256 크기의 고정된 해상도 텐서로 변환
        # 짧은 문장의 픽셀화(격자형 오탐) 방지 및 긴 문장의 패턴 보존
        import torch.nn.functional as F
        # (seq_len, seq_len) -> (1, 1, seq_len, seq_len)
        attn_tensor = pooled_attention.unsqueeze(0).unsqueeze(0)
        fixed_attn = F.interpolate(attn_tensor, size=(256, 256), mode='bicubic', align_corners=False)
        # 다시 2D numpy 배열로 변환
        mean_attention_np = fixed_attn.squeeze().cpu().numpy()
        
        # 4. 히트맵 이미지 생성
        heatmap_img = self._generate_heatmap_image(mean_attention_np)
        
        return heatmap_img, round(quantitative_confidence, 4)

    def _generate_heatmap_image(self, attention_matrix) -> Image.Image:
        """
        Attention 매트릭스를 512x512 해상도의 RGB 히트맵 이미지로 변환합니다.
        """
        # 해상도 512x512 고정을 위해 figsize 설정 (dpi=100일 때 5.12 x 5.12)
        fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)
        
        # 축, 여백 제거
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # 히트맵 렌더링 (컬러맵: hot)
        # 짧은 문장일 경우 nearest 보간법을 쓰면 거대한 블록(격자)처럼 보여 오탐을 유발하므로 bicubic 사용
        # vmax를 1로 고정하지 않고 자동 스케일링하여 짧은 문장에서 특정 단어 하나가 시뻘겋게 튀는 현상 방지
        im = ax.imshow(attention_matrix, cmap='hot', interpolation='bicubic')
        
        # 이미지를 메모리 버퍼에 저장
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        buf.seek(0)
        image = Image.open(buf)
        # RGBA를 RGB로 변환
        if image.mode == 'RGBA':
            image = image.convert('RGB')
            
        # 정확히 512x512로 리사이즈 (토큰 수에 따라 비율이 다를 수 있으므로 강제 조정)
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
        
        return image
