# [The Hallucination Tax of Reinforcement Finetuning](https://arxiv.org/abs/2505.13988)

USC, 2025

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

<details>
<summary>Figure 1: RFT vs RFT+SUM 예시 비교</summary>
<img width="451" alt="Screenshot 2025-07-03 at 16 43 38" src="https://github.com/user-attachments/assets/f3816049-a5b3-448c-b6b8-8e1d1ec54f5b" />
</details>

## 4. SUM 데이터셋 설계와 기준

5가지 Unanswerable 유형

| 유형 | 예시 설명 |
|:----|:-------:|
| 1. 핵심 정보 누락 | 시간, 단위, 조건 없음 |
| 2. 모호한 조건 | "some", "many", 범위 애매 |
| 3. 비현실 조건 | "0분마다", "음수 물건" |
| 4. 문맥 분리 | 질문과 정보 불일치 |
| 5. 질문 삭제 | 질문 자체 제거 |

<details>
<summary>Table 1: SUM 유형별 수정 사례</summary>
<img width="929" alt="Screenshot 2025-07-03 at 16 46 26" src="https://github.com/user-attachments/assets/51a6243c-0c7e-4673-87ac-77037b89c00a" />
</details>

## 5. 실험 설정

- 모델: Qwen2.5 (1.5B / 7B), LLaMA3.1 (8B)
- 학습: PPO 기반 RFT
- 학습 데이터:
  - DeepScaleR (정답 가능한 수학 문제)
  - SUM (비정답 문제, 혼합 비율: 0%, 1%, 10%, 30%, 50%)

## 6. 주요 실험 결과

### RFT만 적용할 경우
- 거절률 0.01 수준 -> 대부분 문제에 억지로 답함

### SUM 10%만 적용해도
- 거절률 대폭 증가 (예: Qwen2.5-7B: 0.01 -> 0.73)

<details>
<summary>Figure 2: 거절률 변화</summary>
<img width="924" alt="Screenshot 2025-07-03 at 16 47 44" src="https://github.com/user-attachments/assets/b799b93d-2b18-4a62-8960-e950fdd692bc" />
</details>

## 7. 정확도 vs 거절률 트레이드오프

| 모델 | 데이터셋 | RFT | RFT+SUM (10%) | 변화 |
|:---:|:------:|----:|--------------:|:----|
| Qwen2.5-7B | GSM8K | 0.90 | 0.88 | -0.02 |
| Qwen2.5-7B | SUM | 0.01 | 0.73 | +0.72 |
| LLaMA-3.1-8B | SelfAware | 0.01 | 0.70 | +0.69 |

<details>
<summary>Table 2: RFT vs RFT+SUM 결과</summary>
<img width="921" alt="Screenshot 2025-07-03 at 16 48 30" src="https://github.com/user-attachments/assets/f0e5e374-6210-46db-9239-2a8aa1e604e5" />
</details>

## 8. RFT 학습 중 거절 학습 곡선

- 거절 능력은 1% -> 10% SUM 비율에서 급증
- 30% 이상은 정답률 감소도 동반

<details>
<summary>Figure 3: Mixing Ratio에 따른 거절/정답률</summary>
<img width="921" alt="Screenshot 2025-07-03 at 16 49 05" src="https://github.com/user-attachments/assets/edd4cb35-3f7b-4645-9846-649ab8103f77" />
</details>

## 9. 정리 및 한계

Epistemic Uncertainty 무시 문제
- RFT 보상 함수는 거절을 보상하지 않음
- “모른다”는 답이 "잘못된 답보다 더 낫다"는 인식 필요

균형 필요
- Reasoning 성능 <-> Trustworthiness 균형 조절
- 향후: Curriculum Learning, Adaptive Reward 등도 탐색 필요

## 10. 결론

- RFT는 성능을 높이지만 자신감 있는 헛소리(hallucination)를 유도할 수 있다.
- SUM 데이터는 LLM이 "모르면 모른다고 말하게" 만든다.
- 정확도를 거의 잃지 않고 신뢰성 향상 가능

> 모델에 “틀릴 땐 거절하라”는 학습을 명시적으로 포함시키자.

## Q&A

### SUM을 수학 외 도메인에 확장

SUM으로 훈련된 모델이 사실 기반 질문(factual QA)에서도 거절 성능이 향상된 것을 보여준다. SelfAware 데이터셋에서 거절률이 0.01에서 0.94로 증가했다. 따라서, 단순히 수학 문제 훈련이 아니라 불확실성 판단을 훈련한 효과라 볼 수 있다.

### Instruction Tuning 모델은 어떤 차이

Instruction-Tuned 모델들이 거절 학습을 더 빠르게 배우고 반응도 더 크다. 예를 들어 Qwen2.5-7B-Instruct는 SUM 데이터 10%만 섞어도 거절률이 0.1에서 0.8 이상으로 급상승했다. 즉, 기존에 사람 말투를 따라 배우던 모델일수록 "모른다고 말하는 법"도 더 잘 배운다.

### Refusal을 평가하는 다른 지표

이 논문에서는 "Refusal Rate"를 단순히 `\boxed{I don't know.}` 문구가 나오는 비율로 평가했지만, 다른 연구에서는: Calibration error, selecive prediction accuracy 등도 사용하고, confidence score나 epistemic uncertainty estimation을 기반으로 평가하기도 한다. 즉, 단순한 string match 외에도 모델의 conf score까지 고려한 평가들이 있다.

---

:bulb: RFT (Reinforcement Fine-Tuning)

RFT는 언어 모델이 더 좋은 답변을 하도록 보상 기준(reward function)을 설정하고, 그에 따라 강화학습을 적용하는 파인튜닝 기법이다. 대표적으로 PPO 알고리즘을 사용하며, 인간 피드백 기반 RFT(RLHF)는 ChatGPT에 적용된 방식이기도 하다.

:bulb: Qwen2.5 / LLaMa-3.1

Qwen은 Alibaba에서, LLaMa는 Meta에서 개발한 대형 언어 모델(LLM) 계열이다. 숫자 (B)은 모델의 크기를 나타내며, 예를 들어 7B는 약 70억 개의 파라미터를 가진 모델을 의미한다.

:bulb: PPO (Proximal Policy Optimization)

PPO는 강화학습에서 널리 사용되는 정책 최적화 알고리즘이다. 학습 중 너무 크게 바뀌는 것을 막아 안정적인 학습을 유도하는 것이 특징이다.

:bulb: Epistemic Uncertainty

Epistemic Uncertainty는 모델이 정보를 충분히 알지 못할 때 생기는 불확실성이다. 이는 "내가 모른다"는 것을 아는 능력으로, 이를 무시할 경우 모델은 자신 없어야 할 상황에서도 확신에 찬 잘못된 답을 하게 된다.

:bulb: Curriculum Learning

Curriculum learning은 쉬운 예저부터 점차 어려운 예제로 학습을 진행하는 방식이다. 사람의 학습 방식과 유사하며, 모델이 안정적으로 개념을 습득하게 도와주는 역할을 한다.

:bulb: Adaptive Reward

Adaptive reward는 상황에 따라 다른 보상값을 주는 방식이다. 예를 들어, 모델이 확신 없을 때 "I don't know"라고 말하면 소정의 보상을 주고, 확신 없이 오답을 낼 경우에는 벌점을 주는 식으로 설정할 수 있다.

:bulb: Instruction-Tuning 모델

Instruction-tuning 모델은 자연어 명령(instruction)을 이해하고 수행할 수 있도록 훈련된 언어 모델이다. 예를 들어, "Summarize this text"나 "Translate this sentence"와 같은 명령에 반응하도록 학습된 모델이며, 지시를 따르는 성향이 강해 "I don't know"와 같은 거절 학습에도 효과적이다.

---

### Reference

- The Hallucination Tax of Reinforcement Finetuning, https://arxiv.org/abs/2505.13988, 2025-06-02-Mon.
- Ask Me Anything: A simple strategy for prompting language models, https://arxiv.org/abs/2210.02441, 2025-07-03-Thu.
  - LLM은 정보가 부족한 질문도 확신 있게 답하려는 경향이 있으며, 질문 형태에 따라 거절률이 달라짐을 보여줌
  - RFT 없이도 prompt만으로 hallucination을 유도하거나 억제할 수 있음을 시사
- Aligning language models to follow instructions, https://openai.com/index/instruction-following/, 2025-07-03-Thu.
  - 인간 피드백을 이용한 RLHF 방식으로 LLM이 지시(instruction)를 따르도록 학습시킨 OpenAI의 대표 연구
  - 이 논문에서 사용하는 RFT의 기반이 된 방식
- Large Language Models as Analogical Reasoners, https://arxiv.org/abs/2310.01714, 2025-07-03-Thu.
  - LLM이 단순 지식 회상이 아닌, 추론 기반 응답을 할 수 있도록 학습 가능함을 실험
  - 단순 RFT가 아닌, 추론에 기반한 보상 설계 필요성에 대한 시사점을 줌
- Measuring and Narrowing the Comprehension Gap in Language Models, https://arxiv.org/abs/2210.03350, 2025-07-03-Thu.
  - LLM이 문제를 이해하고 있는가를 측정하고, 잘못 이해하고도 답하는 현상에 대한 분석
  - hallucination이 comprehension gap에서 비롯된다는 해석 가능
 
