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

### [Meta Llama Models](https://www.llama.com/docs/model-cards-and-prompt-formats/)

* [Llama 4](https://www.llama.com/docs/model-cards-and-prompt-formats/llama4/): Optimized for multimodal understanding, multilingual tasks, coding, tool-calling, and powering agentic systems. The models have a knowledge cutoff of August 2024. Input: Text + up to 5 iamges. Output: Text-only. Arabic, English, French, German, Hindi, Indonesian, Italian, Portuguese, Spanish, Tagalog, Thai, and Vietnames. Image understanding is English-only. Active parameters: 17B.
  * Llama 4 Scout: Number of Experts: 16. Total parameters across active and inactive experts: 109B. Can run on a single GPU: Yes. Maximum Context Length: 10M tokens.
  * Llama 4 Maverick: Number of Experts 128. Total parameters across active and inactive experts: 400B. Can run on a single GPU: No. Maxium Context Length: 1M tokens.

### [Mistral Modles](https://mistral.ai/models)

* [Mistral Small 4](https://mistral.ai/news/mistral-small-4): Apache 2.0 license. Mixture of Experts (MoE): 128 experts, with 4 active per token, enabling efficient scaling and specialization. 119B total parameters, with 6B active parameters per token (8B including embedding and output layers). 256k context window, supporting long-form interactions and document analysis. Configurable reasoning effort: Toggle between fast, low-latency responses and deep, reasoning-intensive outputs. Native multimodality: Accepts both text and image inputs, unlocking use cases from document parsing to visual analysis. Minimum infrastructure: 4x NVIDIA HGX H100, 2x NVIDIA HGX H200, or 1x NVIDIA DGX B200. Recommended setup: 4x NVIDIA HGX H100, 4x NVIDIA HGX H200, or 2x NVIDIA DGX B200 for optimal performance.

### [Google Gemma](https://deepmind.google/models/gemma/)

* [Gemma 3](https://ai.google.dev/gemma/docs/core): Provided with open weights and permit responsible commercial use, allowing you to tune and deploy them in your own projects and applications. Image and text input: Multimodal capabilities let you input images and textto understand and analyze visual data. 128K token context: Significantly large input context for analyzing more data and solving more complex problems. Function calling: Build natural language interfaces for working with programming interfaces. Wide language support: Work in your language or expand your AI application's language capabilities with support for over 140 languages. Developer friendly model sizes: Choose a model size (270M, 1B, 4B, 12B, 27B) and precision level that works best for your task and compute resources.

### [Alibaba Qwen Models](https://qwen.ai/research)

* [Qwen3.5-Max-Preview](https://qwen.ai/blog?id=qwen3.5-max-preview)

### [DeepSeek](https://www.deepseek.com/en/)

* [DeepSeek R1](https://github.com/deepseek-ai/DeepSeek-R1)

### Anthropic

* [Claude Opus 4.7](https://www.anthropic.com/news/claude-opus-4-7)

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
