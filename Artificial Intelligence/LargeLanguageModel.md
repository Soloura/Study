# [Large Language Model (LLM)](https://en.wikipedia.org/wiki/Large_language_model)

A large language model (LLM) is a type of language model notable for its ability to achieve general-purpose language understanding and generation. LLMs acquire these ability by using massive amounts of data to leran billions of parameters during training and consuming large computational resources during their training and operation. LLMs are artificial neural networks (mainly transformers) and are (pre-)trained using self-supervised leraning and semi-supervised learning.

As autoregressive language models, they work by taking an input text and repeatedly predicting the next token or word. Up to 2020, fine tuning was the only way a model could be adapted to be able to accomplish specific tasks. Larger sized models, such as GPT-3, however, can be prompt-engineered to achieve similar results. They are thought to acquire embodied knowledge about syntax, semantics and ontology inherent in human language corpora, but also inaccuracies and biases present in the corpora.

Notable examples include OpenAI's GPT models (e.g., GPT-3.5 and GPT-4, used in ChatGPT), Google's PaLM (used in Bard), and Meta's LLaMa, as well as BLOOM, Erine 3.0 Titan, and Anthropic's Claude 2.

### BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | [arXiv 2019](https://arxiv.org/pdf/1810.04805.pdf) | [GitHub](https://github.com/google-research/bert) | [Article (KR)](https://www.aitimes.kr/news/articleView.html?idxno=13117) | [Wiki](https://en.wikipedia.org/wiki/BERT_(language_model))

BERT is a family of masked-language models published in 2018 by researchers at Google. A 2020 literature survey concluded that "in a little over a year, BERT has become a ubiquitous baseline in NLP experiments counting over 150 research publications analyzing and improving the model."

BERT was originally implemented in the English language at two model sizes: (1) BERT_BASE: 12 encoders with 12 bidirectional self-attention heads totaling 110 million parameters, and (2) BERT_LARGE: 24 encoders with 16 bidirectional self-attentino heads totaling 340 million parameters. Both models were pre-trained on the Toronto BookCorpus (800M words) and English Wikipedia (2,500M words).

### GPT-1 | [OpenAI](https://openai.com/blog/language-unsupervised/) | [Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | [GitHub](https://github.com/openai/finetune-transformer-lm)

Improving Language Understanding with Generative Pre-Training

### GPT-2 | [OpenAI](https://openai.com/blog/better-language-models/) | [Follow-up Post](https://openai.com/blog/gpt-2-6-month-follow-up/) | [Final Post](https://www.openai.com/blog/gpt-2-1-5b-release/) | [Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) | [GitHub](https://github.com/openai/gpt-2)

Language Models are Unsupervised Multitask Learners

### GPT-3 - A set of modesl that can understand and generate natural language | [OpenAI API](https://openai.com/blog/openai-api/) | [Paper](https://arxiv.org/pdf/2005.14165.pdf) | [GitHub](https://github.com/openai/gpt-3)

Language Models are Few-Shot Learners

### InstructGPT: Aligning Language Models to Follow Instructions | [OpenAI](https://openai.com/blog/instruction-following/) | [arXiv](https://arxiv.org/abs/2203.02155)

### ChatGPT: Optimizing Language Models for Dialoge | [OpenAI](https://openai.com/blog/chatgpt/) | [chat](https://chat.openai.com/chat)

ChatGPT is a variant of the popular GPT (Generative Pre-training Transformer) language model that was specifically designed for chatbot applications. Like other GPT models, ChatGPT uses machine learning techniques to generate human-like text based on a given prompt. However, ChatGPT has been trained on a large dataset of conversations and is optimized for generating responses that are appropriate for use in chatbot scenarios. ChatGPT is able to generate responses that are coherent, contextually relevant, and appropriate for the tone of the conversation. It can be used to power chatbots for customer service, e-commerce, entertainment, and many other applications.

### Linguistic Model | [ScienceDirect](https://www.sciencedirect.com/topics/computer-science/linguistic-model)

Linguistic models deal with statements as they are used to express meanings. While fact models focus on simple operational meanings within a given conceptual commitment (existence of objects, discernablility of objects as representatives of certain concepts, characteristics of individual objects and relationships between objects), linguistic models focus on meanings corresponding to the conceptual commitment itself, regardless of the actual objects such as what are the charcteristics of concepts, how concepts are defined in terms of other concepts, and what relationships between concepts are necessary, permissible or obligatory. Thus statements in fact models describe individual views while statements in linguistic models describe viewpoints. Linguistic models focus at the definitions of new meanings, therefore they facilitate unconstrained communications where new meanings are defined on-the-fly. The flexibility of unrestricted linguistic communication comes with a price: the intended meaning needs to be unraveled from the expression, and the more complex the meaning, the more complex is the unraveling process. The key challenge of unrestricted linguistic communication is not so much in parsing a natural language sentence (which can be quite complex) but rather in dealing with complex meanings that are defined on-the-fly by some sentences and are used in the follow-up sentences. For example, we can say: “Remember the nuts-and-bolts illustrations from the previous chapter? Did you notice that there are a total of 35 nuts visible in these figures?” While humans routinely deal with the dynamic updates to the conceptual commitment, it is a barrier for machine-to-machine information exchanges.

### GPT-4 - A set of models can understand as well as generate natural language or code | [OpenAI](https://openai.com/product/gpt-4) | [arXiv 2023](https://arxiv.org/abs/2303.08774)

---

### [Large Concept Models: Language Modeling in a Sentence Representation Space](https://ai.meta.com/research/publications/large-concept-models-language-modeling-in-a-sentence-representation-space/) | [arXiv](https://arxiv.org/abs/2412.08821)

This paper introduces a novel architectur called the Large Concept Model (LCM), which operates at a higher-level semantic representation, termed "concepts", rather than the traditional token-level processing used by LLMs. Concepts are language- and modality-agnostic and represent broader ideas or actions.

As a proof of feasibility, the study assumes that concepts correspond to sentences and leverages the SONAR sentence embedding space, which supports 200 languages across text and speech. The LCM is trained for autoregressive sentence prediction in this embedding space using methods like MSE regression, diffusion-based generation, and quantized SONAR models.

Experiments were conducted with 1.6B parameter models trained on 1.3T tokens, and later scaled to 7B parameters with 7.7T tokens. The model was tested on summarization and summary expansion tasks, showing strong zero-shot generalization across multiple languages, outperforming existing LLMs of the same size. The training code is freely available.

---

## Model

### OpenAI

* [GPT-2 1.5B](https://openai.com/index/gpt-2-1-5b-release/)
* [GPT-4o mini](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/)
* [GPT-5.4](https://openai.com/index/introducing-gpt-5-4/)
* [GPT-5.3-Codex](https://openai.com/index/introducing-gpt-5-3-codex/)

### Google [Gemma](https://deepmind.google/models/gemma/)

* [Gemma 3](https://ai.google.dev/gemma/docs/core): Provided with open weights and permit responsible commercial use, allowing you to tune and deploy them in your own projects and applications. Image and text input: Multimodal capabilities let you input images and textto understand and analyze visual data. 128K token context: Significantly large input context for analyzing more data and solving more complex problems. Function calling: Build natural language interfaces for working with programming interfaces. Wide language support: Work in your language or expand your AI application's language capabilities with support for over 140 languages. Developer friendly model sizes: Choose a model size (270M, 1B, 4B, 12B, 27B) and precision level that works best for your task and compute resources.
* [Gemma 4](https://deepmind.google/models/gemma/gemma-4/): Gemma 4 model family spans three distinct architectures tailored for specific hardware requirements: Small Sizes: 2B and 4B effective parameter models built for ultra-mobile, edge, and browser deployment (e.g., Pixel, Chrome). Dense: A powerful 31B parameter dense model that bridges the gap between server-grade performance and local execution. Mixture-of-Experts: A highly efficient 26B MoE model designed for high-throughput, advanced reasoning.

### Anthropic Claude [Haiku](https://www.anthropic.com/claude/haiku) [Sonnet](https://www.anthropic.com/claude/sonnet) [Opus](https://www.anthropic.com/claude/opus)

* [Claude Haiku 4.5](https://www.anthropic.com/news/claude-haiku-4-5)
* [Claude Sonnet 4.6](https://www.anthropic.com/news/claude-sonnet-4-6)
* [Claude Opus 4.6](https://www.anthropic.com/news/claude-opus-4-6)
* [Claude Opus 4.7](https://www.anthropic.com/news/claude-opus-4-7)

### Meta [Llama Models](https://www.llama.com/docs/model-cards-and-prompt-formats/)

* [Llama 4](https://www.llama.com/docs/model-cards-and-prompt-formats/llama4/): Optimized for multimodal understanding, multilingual tasks, coding, tool-calling, and powering agentic systems. The models have a knowledge cutoff of August 2024. Input: Text + up to 5 iamges. Output: Text-only. Arabic, English, French, German, Hindi, Indonesian, Italian, Portuguese, Spanish, Tagalog, Thai, and Vietnames. Image understanding is English-only. Active parameters: 17B.
  * Llama 4 Scout: Number of Experts: 16. Total parameters across active and inactive experts: 109B. Can run on a single GPU: Yes. Maximum Context Length: 10M tokens.
  * Llama 4 Maverick: Number of Experts 128. Total parameters across active and inactive experts: 400B. Can run on a single GPU: No. Maxium Context Length: 1M tokens.

### Mistral [Modles](https://mistral.ai/models)

* [Mistral 3](https://mistral.ai/news/mistral-3)
* [Mistral OCR 3](https://mistral.ai/news/mistral-ocr-3)
* [Mistral Small 4](https://mistral.ai/news/mistral-small-4): Apache 2.0 license. Mixture of Experts (MoE): 128 experts, with 4 active per token, enabling efficient scaling and specialization. 119B total parameters, with 6B active parameters per token (8B including embedding and output layers). 256k context window, supporting long-form interactions and document analysis. Configurable reasoning effort: Toggle between fast, low-latency responses and deep, reasoning-intensive outputs. Native multimodality: Accepts both text and image inputs, unlocking use cases from document parsing to visual analysis. Minimum infrastructure: 4x NVIDIA HGX H100, 2x NVIDIA HGX H200, or 1x NVIDIA DGX B200. Recommended setup: 4x NVIDIA HGX H100, 4x NVIDIA HGX H200, or 2x NVIDIA DGX B200 for optimal performance.

### Alibaba [Qwen Models](https://qwen.ai/research)

* [Qwen3.5](https://qwen.ai/blog?id=qwen3.5)
* [Qwen3.5-Max-Preview](https://qwen.ai/blog?id=qwen3.5-max-preview)
* [Qwen3.6](https://github.com/QwenLM/Qwen3.6): the large language model series developed by Qwen team, Alibaba Group.
* [Qwen3.6-35B-A3B](https://qwen.ai/blog?id=qwen3.6-35b-a3b): a fully open-source MoE model (35B total / 3B active), featuring: exceptional agentic coding capability competitive with much larger models strong multimodal perception and reasoning ability

### [DeepSeek](https://www.deepseek.com/en/)

* [DeepSeek V3](https://github.com/deepseek-ai/DeepSeek-V3)
* [DeepSeek R1](https://github.com/deepseek-ai/DeepSeek-R1)

---

* OpenAI
  * GPT-2 1,5B (2019-11-05)
  * GPT-4o mini (2024-07-18)
  * GPT-5 (2025-08-07)
  * GPT-5.4 (2025-03-05)
  * GPT-5.3-Codex (2026-02-05)
* Google DeepMind
  * Gemma 3 270M (2025-08-14)
  * Gemma 4 (2026-04-02)
* Anthropic
  * Claude Haiku 4.5 (2025-10-15)
  * Claude Opus 4.6 (2026-02-05)
  * Claude Sonnet 4.6 (2026-02-17)
  * Cluade Opus 4.7 (2026-04-16)
* Meta
  * Llama 4
* Mistral AI
  * Mistral 3 (2025-12-02)
  * Mistral OCR 3 (2025-12-17)
  * Mistral Small 4 (2026-03-16)
* Alibaba Cloud
  * Qwen3.5 (2026-02-16)
  * Qwen3.6-Plus (2026-04-02)
  * Qwne3.6-35B-A3B (2026-04-15)
* DeekSeek
  * DeepSeek V3 (2024-12-26)
  * DeekSeek R1 (2025-01-20)
  * DeepSeek V3.1 (2025-08-21)
 
---

### Skill & Plugin

* [Superpowers](https://github.com/obra/superpowers): Superpowers is a complete software development methodology for your coding agents, built on top of a set of composable skills and some initial instructions that make sure your agent uses them.
* [Microsoft Docs](https://claude.com/plugins/microsoft-docs)

---

### Coding Agent & Agentic Coding

* [Cursor](https://cursor.com/agents): Cursor is an AI editor and coding agent. Use it to understand your codebase, plan and build features, fix bugs, review changes, and work with the tools you already use.
* [Amazon Q Developer](https://aws.amazon.com/q/): Amazon Q Developer is the most capable generative AI-powered assistant for building, operating, and transforming software, with advanced capabilities for managing data and AI/ML.
* [Kiro](https://kiro.dev/): Kiro is an agentic IDE that helps you do your best work with features such as specs, steering, and hooks.
* [Codex](https://openai.com/ko-KR/codex/): A coding agent that helps you build and ship with AI—powered by ChatGPT.
* [Claude Code](https://claude.com/product/claude-code): Claude Code is an agentic coding tool that reads your codebase, edits files, runs commands, and integrates with your development tools. Available in your terminal, IDE, desktop app, and browser.

---

### Pricing

SUNO
* Pro KRW12,000/15,000 USD8/10

GPT
* Free:  KRW       0 / month ; Intelligence for everyday tasks
* Go  :  KRW  13,000 / month ; Keep chatting with expanded access
* Plus:  KRW  29,000 / month ; Do more with advanced intelligence
* Pro :  KRW 159,000 / month ; Maimize your productivity

Google AI
* Plus : KRW   7,500 / month
* Pro  : KRW  29,000 / month
* Ultra: KRW 119,000 / month

Claude
* Free   :  USD   0 / month ; Free for everyone
* ~~Pro    :  USD  17 / month ; with annual subscription discount USD200 billed up front~~
* Pro    :  USD  19 / month ;(17% off - year)
* Pro    :  USD  22 / month ;
* ~~Pro    :  USD  20 / month ; if billed monthly~~
* ~~Max 5x :  USD 100 / month~~
* ~~Max 20x:  USD 200 / month~~
* Max    :  USD 110 / month

---

# [AI 로컬 구동 및 학습 인프라 핵심 아키텍처 요약]

## 1. 인퍼런스(구동) VRAM 소모 메커니즘
* **가중치(Weights)의 기본 점유**: 파라미터 수(B) × 데이터 타입 바이트(Byte) 크기로 고정 점유됨.
  * FP16(원본): 1B 파라미터당 2GB 소모 (예: Gemma 4 12B Unified = 약 24~25GB VRAM)
  * Q4(4비트 양자화): 1B 파라미터당 약 0.5GB 소모 (예: 12B 모델 = 약 7~8GB VRAM)
* **런타임 오버헤드**: 실제 챗봇 구동 시 Context Window(입출력 텍스트) 길이에 따른 KV 캐시 누적 및 CUDA 오버헤드로 수 GB의 유동적 공간이 추가 요구됨.

## 2. 학습(Training) 리소스 폭증 원인 (1B 모델 FP16 기준)
* **인퍼런스(구동)**: 모델 가중치(2GB)만 상주하면 구동 가능.
* **트레이닝(학습)**: 가중치만으로 불가능하며, 아래 4가지 요소 결합으로 **최소 8~10배(16~20GB+) 리소스** 폭증.
  1. 모델 가중치 (Model Weights): FP16 정밀도 기준 2GB
  2. 미분값 (Gradients): 오차 역전파 계산용 2GB
  3. 최적화 상태 (Optimizer States): Adam 알고리즘의 과거 기록 보존(FP32 정밀도)을 위해 파라미터당 8바이트 = 8GB
  4. 활성화 값 (Activation): 문장 길이에 따른 순방향 연산 중간 결과 저장용 가변 4~8GB+

## 3. 리소스 한계 극복 기술: LoRA & QLoRA
* **LoRA (Low-Rank Adaptation)**: 원본 가중치는 고정(Freeze)하고, 미세 조정을 위한 1% 미만의 소규모 '학습용 어댑터(행렬)'만 추가하여 미분 및 옵티마이저 메모리 소모량을 99% 절감.
* **QLoRA (Quantized LoRA)**: 고정하는 원본 가중치마저 4비트로 양자화 압축하여 VRAM에 로드한 뒤 LoRA를 적용. 일반 소비자용 GPU(RTX 3090/4090) 1~2대로 거대 모델 학습을 가능하게 만드는 핵심 돌파구.
* **도메인별 활용**:
  * 텍스트: 특정 도메인 전문 지식 주입 및 출력 형식(JSON 등) 강제화
  * 이미지: 고유 캐릭터 외모 유지, 특정 작가의 화풍 및 연출 스타일 고정
  * 비디오: 카메라 워킹(드론 샷 등), 비디오 고유 스타일, 프레임 간 캐릭터 연속성 유지

## 4. 인프라 운영 생태계 (빅테크 vs 일반 기업/학교)
* **사전 학습 (Pre-training)**: 아무것도 모르는 AI에게 세상의 지식을 가르쳐 '베이스 모델(Base Model)'을 만드는 단계로, LoRA 없이 수만 대의 GPU로 100% 전체 학습 진행 (빅테크 영역).
* **미세 조정 (Fine-tuning)**: 완성된 베이스 모델을 가져와 특정 업무를 교육하는 단계로, 비용 절감을 위해 일반 기업/대학은 LoRA/QLoRA를 필수 채택.
* **동시 접속자(Concurrent Users) 처리**: vLLM 등의 배치 기술로 단일 GPU 노드당 동시 16~64명(가입자 기준 1,000~3,000명)을 소화하며, 유저 증가 시 모델 복사본(Replica)을 늘려 로드 밸런싱 수행.

## 5. MoE(Mixture of Experts) 모델의 리소스 반전 구조
* **연산(Compute) 리소스 = 활성 파라미터(Active)**: 라우터를 통해 질문에 맞는 2~3개의 전문가 모델만 깨워 연산하므로 연산 비용이 낮고 답변 속도가 극도로 빠름.
* **메모리(VRAM) 리소스 = 전체 파라미터(Total)**: 다음 질문이 무엇일지 예측할 수 없으므로, 전체 전문가 가중치(예: DeepSeek V3 기준 671B = 약 1.3TB VRAM)가 메모리에 상시 대기하고 있어야 함. 
* **결론**: VRAM 구축 비용(초기 인프라 돈)은 무지막지하게 들지만, 한번 세팅하면 운영 전기세(연산 비용)와 속도 측면에서 압도적 이득을 보는 구조.

# [최신 프론티어 AI 모델별 파라미터 및 리소스 상세 가이드]

## 1. 초거대급 프론티어 플래그십 (GPT-5.4, Gemini 3.1, Claude 4.6 Opus)
빅테크 독점 모델로 세부 수치는 업계 추정치 및 유출 스펙 기준입니다. 단일 서버가 아닌 초대형 분산 인프라 레이아웃이 필수적입니다.

### GPT-5.4 (OpenAI)
* **전체 파라미터 (Total)**: 약 2.0T ~ 2.5T (2조~2.5조 개, 초대형 MoE 구조)
* **활성 파라미터 (Active)**: 토큰당 약 200B ~ 300B 활성화
* **인퍼런스 최소 VRAM**: 기본 약 4,000GB+ VRAM 요구 (H100 80GB × 최소 64대 이상의 텐서 병렬화 클러스터 1세트 상시 대기)
* **풀 트레이닝 예상 리소스**: B200 / H100 등 최고 사양 GPU 20,000 ~ 30,000대 이상이 결합된 인프라에서 수개월간 수행

### Gemini 3.1 Ultra / Pro (Google)
* **전체 파라미터 (Total)**: 약 1.5T ~ 2.0T (MoE 구조)
* **활성 파라미터 (Active)**: 토큰당 약 49B (Pro 기준) ~ 100B+ (Ultra 기준) 활성화
* **인퍼런스 최소 VRAM**: 약 3,000GB+ VRAM 버퍼 세팅 (1M~2M+ 초대용량 컨텍스트 윈도우 지원을 위한 대규모 KV 캐시 추가 공간 포함)
* **풀 트레이닝 예상 리소스**: 구글 자체 AI 가속기 인프라인 TPU v5p 및 v6 초대형 팟(Pod) 수천 대를 다중 노드로 결합하여 연산

### Claude 4.6 Opus (Anthropic)
* **전체 파라미터 (Total)**: 약 1.8T ~ 2.2T (추론 능력 극대화 MoE 구조)
* **활성 파라미터 (Active)**: 토큰당 약 150B ~ 200B 활성화
* **인퍼런스 최소 VRAM**: 약 3,600GB+ VRAM 수준 요구 (H100 80GB 가속기 48대~64대가 동시 연산 병렬화 서빙)
* **풀 트레이닝 예상 리소스**: AWS(아마존) 및 자체 인프라의 차세대 Trainium2 또는 H100 GPU 15,000대 이상 규모의 초대형 팟 활용

---

## 2. 고효율 플래그십 및 오픈웨이트 (Claude 4.6 Sonnet, DeepSeek V3, Grok 4.2)
비용 효율과 토큰 처리 속도의 밸런스를 맞춰 실무 및 상용 에이전트 서비스 백엔드로 가장 선호되는 체급입니다.

### Claude 4.6 Sonnet (Anthropic)
* **전체 파라미터 (Total)**: 약 400B ~ 500B (가밀도 MoE 구조)
* **활성 파라미터 (Active)**: 토큰당 약 50B ~ 80B 활성화
* **인퍼런스 최소 VRAM**: 원본 FP16 기준 약 800GB+ VRAM 필요 (H100 80GB 8대 장착된 단일 노드 서버 1개로 독립 마지노선 구축 가능)
* **풀 트레이닝 예상 리소스**: 고성능 인프라 클러스터 GPU 약 5,000 ~ 8,000대 규모 백엔드에서 훈련 진행

### DeepSeek V3 (DeepSeek)
* **전체 파라미터 (Total)**: 671B (MLA 및 보조 손실 없는 획기적 MoE 구조)
* **활성 파라미터 (Active)**: 토큰당 정확히 37B 활성화 (공식 백서 명세 기준)
* **인퍼런스 최소 VRAM**: 가중치 안착을 위해 FP16 기준 최소 1,342GB VRAM 필수 (H100 80GB × 16대 노드 묶음, 또는 INT4 양자화 서빙 시 H100 8대 단일 노드 1세트)
* **풀 트레이닝 예상 리소스**: H800 GPU 2,000대 규모 클러스터에서 총 278.8만 GPU 시간 소모 (가성비 알고리즘 덕분에 압도적으로 적은 하드웨어로 조율)

### Grok 4.2 (xAI)
* **전체 파라미터 (Total)**: 약 300B ~ 400B (MoE 구조)
* **활성 파라미터 (Active)**: 토큰당 약 40B ~ 50B 활성화
* **인퍼런스 최소 VRAM**: 원본 구동 시 약 600GB ~ 800GB VRAM 수준 (H100 8대 구성 사내 서버 1노드로 안전 구동 타협선)
* **풀 트레이닝 예상 리소스**: 일론 머스크 멤피스 'Colossus' 단지의 초고밀도 H100 10만 대 슈퍼컴퓨터 단지 일부 인프라 가동

---

## 3. 기업/연구실 로컬 가성비 체급 (Qwen 3.6-Plus, Claude 4.5 Haiku)
사내 보안망 구축, 로컬 파인튜닝 실험, 다중 소비자용 GPU 분산 서빙이 현실적으로 허용되는 최적의 경량 라인업입니다.

### Qwen 3.6-Plus (Alibaba)
* **전체 파라미터 (Total)**: 약 70B ~ 110B (고밀도 Dense 혹은 준대형 MoE 구조)
* **활성 파라미터 (Active)**: 구조에 따라 전체 점유 또는 약 30B 내외 활성화
* **인퍼런스 최소 VRAM**: 
  * 원본 FP16 기준: 약 150GB ~ 220GB VRAM (H100 2~4대 매칭)
  * Q4 4비트 양자화: 약 40GB ~ 60GB 수준 점유 (일반 소비자용 워크스테이션 RTX 3090/4090 24GB × 2~3장을 묶어 로컬 독립 구동 가능)
* **풀 트레이닝 예상 리소스**: 대기업 자체 클라우드 인프라의 GPU 수백 대 수준에서 수일 만에 풀 트레이닝 및 완전 미세조정 정복 가능

### Claude 4.5 Haiku (Anthropic)
* **전체 파라미터 (Total)**: 약 20B ~ 30B (초경량 Dense 고밀도 모델)
* **활성 파라미터 (Active)**: Dense 구조로 매 토큰당 20B ~ 30B 전체 파라미터 연산 참여
* **인퍼런스 최소 VRAM**: 
  * 원본 FP16 기준: 약 40GB ~ 60GB VRAM 요구 (엔터프라이즈 단일 카드 RTX 6000 Ada 48GB 1장 혹은 RTX 4090 2~3장 분산)
  * 양자화 버전 구동 시: 단 1장의 RTX 4090(24GB) 단독 탑재 PC 환경에서도 원활하고 기민하게 초고속 서빙 가능
* **풀 트레이닝 예상 리소스**: GPU 수십 대 미만의 소규모 대학 연구실, 일반 중소기업 태스크포스(TF) 팀에서도 예산 범위 내 자체 Full Fine-tuning 시도 가능



---

### Reference
- Large Language Model Wiki, https://en.wikipedia.org/wiki/Large_language_model, 2023-10-23-Mon.
- GPT-3 Article KR, https://www.aitimes.kr/news/articleView.html?idxno=17370, 2020-09-29-Tue.
- GPT-1 Review Kor, https://www.quantumdl.com/entry/12%EC%A3%BC%EC%B0%A81-Improving-Language-Understanding-by-Generative-Pre-Training, 2020-10-08-Thu.
- What is 'BERT'? Article KR, http://www.aitimes.kr/news/articleView.html?idxno=13117, 2020-11-30-Mon.
- GPT-3 License Article KR, http://www.aitimes.kr/news/articleView.html?idxno=17893, 2020-11-30-Mon.
- OpenAI ChatGPT, https://openai.com/blog/chatgpt/, 2022-12-09-Fri.
- OpenAI ChatGPT Try, https://chat.openai.com/auth/login, 2022-12-09-Fri.
- OpenAI InstructGPT, https://openai.com/blog/instruction-following/, 2022-12-09-Fri.
- OpenAI InstructGPT arXiv, https://arxiv.org/abs/2203.02155, 2022-12-09-Fri.
- GPT-4 OpenAI, https://openai.com/product/gpt-4, 2023-03-16-Thu.
- BERT Article KR, https://www.aitimes.kr/news/articleView.html?idxno=13117, 2023-03-21-Tue.
- BERT arXiv, https://arxiv.org/pdf/1810.04805.pdf, 2023-03-21-Tue.
- BERT GitHub, https://github.com/google-research/bert, 2023-03-21-Tue.
- BERT Wiki, https://en.wikipedia.org/wiki/BERT_(language_model), 2023-03-21-Tue.
- LCM Blog KR, https://discuss.pytorch.kr/t/lcm-large-concept-models-meta-ai/5744, 2025-03-17-Mon.
- Large Concept Models Meta, https://ai.meta.com/research/publications/large-concept-models-language-modeling-in-a-sentence-representation-space/, 2025-03-17-Mon.
- Large Concept Models arXiv, https://arxiv.org/abs/2412.08821, 2025-03-17-Mon.
- Qwen Models, https://qwen.ai/research, 2026-03-28-Sat.
- Mistral AI Models, https://mistral.ai/models, 2026-03-28-Sat.
- Meta Llama Models, https://www.llama.com/docs/model-cards-and-prompt-formats/, 2026-03-28-Sat.
- Llama 4, https://www.llama.com/docs/model-cards-and-prompt-formats/llama4/, 2026-03-28-Sat.
- Mistral 4 Small, https://mistral.ai/news/mistral-small-4, 2026-03-28-Sat.
- Google DeepMind Models, https://deepmind.google/models, 2026-03-28-Sat.
- Google Gemma Models, https://deepmind.google/models/gemma/, 2026-03-28-Sat.
- Google Gemma 3 Model Overview, https://ai.google.dev/gemma/docs/core, 2026-03-28-Sat.
- Alibaba Qwen Models, https://qwen.ai/research, 2026-03-28-Sat.
- Alibaba Qwen3.5-Max-Preview, https://qwen.ai/blog?id=qwen3.5-max-preview, 2026-03-28-Sat.
- DeepSeek, https://www.deepseek.com/en/, 2026-03-28-Sat.
- DeepSeek R1, https://github.com/deepseek-ai/DeepSeek-R1, 2026-03-28-Sat.
- Claude Opus 4.7, https://www.anthropic.com/news/claude-opus-4-7, 2026-04-18-Sat.
- Gemma 4, https://deepmind.google/models/gemma/gemma-4/, 2026-04-18-Sat.
- Gemma 4, https://ai.google.dev/gemma/docs/core, 2026-04-18-Sat.
- Qwen3.6, https://github.com/QwenLM/Qwen3.6, 2026-04-18-Sat.
- Gemma, https://ai.google.dev/gemma/docs, 2026-04-18-Sat.
- Qwen3.6-Plus, https://qwen.ai/blog?id=qwen3.6, 2026-04-18-Sat.
- Qwen3.6-35B-A3B, https://qwen.ai/blog?id=qwen3.6-35b-a3b, 2026-04-18-Sat.
- Cluade Opus, https://www.anthropic.com/claude/opus, 2026-04-18-Sat.
- Claude Opus 4.6, https://www.anthropic.com/news/claude-opus-4-6, 2026-04-18-Sat.
- Claude Haiku, https://www.anthropic.com/claude/haiku, 2026-04-18-Sat.
- Claude Haiku 4.5, https://www.anthropic.com/news/claude-haiku-4-5, 2026-04-18-Sat.
- Claude Sonnet, https://www.anthropic.com/claude/sonnet, 2026-04-18-Sat.
- Claude Sonnet 4.6, https://www.anthropic.com/news/claude-sonnet-4-6, 2026-04-18-Sat.
- Gemma 3, https://deepmind.google/models/gemma/gemma-3/, 2026-04-18-Sat.
- Gemma 3 270M, https://developers.googleblog.com/en/introducing-gemma-3-270m/, 2026-04-18-Sat.
- GTP-5.4, https://openai.com/index/introducing-gpt-5-4/, 2026-04-18-Sat.
- GPT-5.3-Codex, https://openai.com/index/introducing-gpt-5-3-codex/, 2026-04-18-Sat.
- Qwen3.5, https://qwen.ai/blog?id=qwen3.5, 2026-04-18-Sat.
- Mistral 3, https://mistral.ai/news/mistral-3, 2026-04-18-Sat.
- Mistral OCR 3, https://mistral.ai/news/mistral-ocr-3, 2026-04-18-Sat.
- GPT-2 1.5B, https://openai.com/index/gpt-2-1-5b-release/, 2026-04-18-Sat.
- GPT-5, https://openai.com/index/introducing-gpt-5/, 2026-04-18-Sat.
- Codex, https://openai.com/index/introducing-codex/, 2026-04-18-Sat.
- GPT-4o mini, https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/, 2026-04-18-Sat.
- Whisper, https://openai.com/index/whisper/, 2026-04-18-Sat.
- DeepSeek V3, https://github.com/deepseek-ai/DeepSeek-V3, 2026-04-19-Sun.
- Superpowers, https://github.com/obra/superpowers, 2026-04-20-Mon.
- Superpowers, https://claude.com/plugins/superpowers, 2026-04-20-Mon.
- Microsoft Docs, https://claude.com/plugins/microsoft-docs, 2026-04-20-Mon.
- Kiro, https://kiro.dev/, 2026-04-21-Tue.
- Kiro Docs, https://kiro.dev/docs/, 2026-04-21-Tue.
- Cursor, https://cursor.com/agents, 2026-04-21-Tue.
- Cursor Docs, https://cursor.com/docs, 2026-04-21-Tue.
- Amazon Q, https://aws.amazon.com/q/, 2026-04-21-Tue.
- Claude Code, https://claude.com/product/claude-code, 2026-04-21-Tue.
- Claude Code, https://code.claude.com/docs/en/overview, 2026-04-21-Tue.
- Codex, https://openai.com/ko-KR/codex/, 2026-04-21-Tue.
- Claude Pricing, https://claude.com/pricing, 2026-04-24-Fri.
- ChatGPT Pricing, https://chatgpt.com/pricing/, 2026-04-24-Fri.
- Azure Architecture, https://learn.microsoft.com/en-us/azure/architecture/, 2026-04-25-Sat.
- Azure AI Agent Orchestration Patterns, https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns, 2026-04-25-Sat.
- Azure GenAIOps, https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/genaiops-for-mlops, 2026-04-25-Sat.
- Azure AI Architecture Design, https://learn.microsoft.com/en-us/azure/architecture/ai-ml/, 2026-04-25-Sat.
- Claude Code Agent Teams Subagent Blog KR, https://goddaehee.tistory.com/517, 2026-05-09-Sat.
- ChatGPT Pricing, https://chatgpt.com/#pricing, 2026-07-24-Fri.
- Gemini Pricing, https://one.google.com/ai, 2026-07-24-Fri.
- Claude Pricing, https://claude.ai/upgrade, 2026-07-24-Fri.

Gemma 4 로컬 구동 가이드 및 LM Studio 모델 저장소 (2025-12-10)
vLLM 고성능 로컬 서빙 엔진 공식 깃허브 아키텍처 (2026-03-15)
TGI (Text Generation Inference) 분산 대규모 서빙 문서 (2026-02-20)
DeepSeek V3 기술 백서 - 671B MoE 구조 및 학습 리소스 명세 (2024-12-26)
Hugging Face 분산 학습 및 가중치 병렬화(Tensor Parallelism) 가이드 백서 (2022-01-05)
AceCloud - 2026 최신 오픈소스 LLM 트렌드 및 모델 하드웨어 벤치마크 가이드 (2026-07-16)
DeepSeek AI 공식 저장소 - V3 모델 아키텍처 가중치 기술 백서 (2025-06-27)
arXiv - DeepSeek-V3 기술 리포트: MLA 및 가성비 MoE 훈련 스펙 상세 (2024-12-27)
Google AI for Developers - Gemma 4 핵심 오픈소스 라인업 공식 오버뷰 및 요구 메모리 (2026-07-08)
NVIDIA NeMo Framework - DeepSeek V3 파인튜닝 레시피 및 가속화 명세 (2026-01-13)
OpenAI API Docs - 최신 플래그십 GPT-5.4 릴리즈 사양 및 컨텍스트 가격 명세 (2026-03-17)
