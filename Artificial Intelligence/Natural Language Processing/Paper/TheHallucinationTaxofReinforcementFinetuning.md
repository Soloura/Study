# The Hallucination Tax of Reinforcement Finetuning

### USC, 2025

> RFT로 추론 성능이 향상되었지만, 환각 문제가 심화된다?

## 1. 서론

- RFT (Reinforcement Fine-tuning): LLM의 추론 능력 향상을 위한 표준적 기법
- 하지만:
  - 모호하거나 답이 없는 문제(unanswerable)에서도
  - 자신감 있게 틀린 답을 반환(hallucination)

> 이 현상을 저자들은 Hallucination Tax라고 부름

-> 모델은 모르면 "모른다"고 말할 수 있는가?

## 2. 핵심 기여 요약

- Hallucination Tax: RFT가 거절 능력을 억제하여 과도한 환각을 유발함
- SUM (Synthetic Unanswerable Math): 인위적 비답변 수학 문제 데이터셋 제안
- 10% SUM 데이터만으로도 거절 능력 복원 가능
- 정답률 손실은 거의 없음

## 3. e.g., RFT 전후 비교

### Unanswerable 문제 예시

```Text
질문: 1부터 시작하는 연속된 양의 정수 중 하나가 지워졌다. 남은 수들의 평균이 35 + 7/17이다. 어떤 수가 지워졌을까?
```

| 모델 | 생각하기(thought) | 출력 |
|:----|:----------------|:----|
| RFT only | 불완전한 정보를 기반으로 계산을 강행 | `\boxed{37}` (X) |
| RFT + SUM | 정보 부족을 인식하고 포기 | `\boxed{I don't know}` (O) |

**[Placeholder: Figure - RFT vs RFT+SUM 예시 비교 다이어그램]**

## 4. SUM 데이터셋 설계와 기준

5가지 Unanswerable 유형

| 유형 | 예시 설명 |
|:----|:-------:|
| 1. 핵심 정보 누락 | 시간, 단위, 조건 없음 |
| 2. 모호한 조건 | "some", "many", 범위 애매 |
| 3. 비현실 조건 | "0분마다", "음수 물건" |
| 4. 문맥 분리 | 질문과 정보 불일치 |
| 5. 질문 삭제 | 질문 자체 제거 |

**[Placeholder: Table - SUM 데이터셋 수정 예시 표]**

## 5. 실험 설정

- 모델: Qwen2.5 (1.5B / 7B), LLaMA3.1 (8B)
- 학습: PPO 기반 RFT
- 학습 데이터:
  - DeepScaleR (정답 가능한 수학 문제)
  - SUM (비정답 문제, 혼합 비율: 0%, 1%, 10%, 30%, 50%)

**[Placeholder: Figure - PPO RFT 학습 구성도 (SUM 혼합 비율 포함)]**
