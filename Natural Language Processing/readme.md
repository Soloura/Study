# Natural Language Processing
## Word Embedding
### Word2Vec | Word Embedding
### GloVe | Word Embedding
### fastText | Word Embedding | [Homepage](https://research.fb.com/blog/2016/08/fasttext/)
### NPLN(Neural Probabilistic Language Model)
### SVD(Singular Value Decomposion)
### PCA(Principal Component Analysis)
### LSA(Latent Sematic Analysis)

### Bidirectional Encoder Representations from Transformers(BERT)

2018년 11월에 Google이 공개한 언어모델 BERT는 일부 성능 평가에서 인간보다 더 높은 정확도를 보이며 2018년 말, 상위 딥러닝 모델이다. 또한 BERT는 언어표현 사전학습의 새로운 방법으로 그 의미는 큰 텍스트 코퍼스(Wikipedia와 같은)를 이용하여 범용목적의 언어 이해(Language Understanding) 모델을 훈련시키는 것과 그 모델에 관심 있는 실제의 자연 언어 처리 태스트(질문, 응답 등)에 적용하는 것이다. 특히 BERT는 종래보다 우수한 성능을 발휘한다. BERT는 자연어 처리 태스크를 교육 없이 양방향으로 이용해 훈련되고 있다는 것을 의미한다. 이것은 Web 상에서 막대한 양의 보통 텍스트 데이터가 여러 언어로 이용 가능하기 때문에 중요한 특징으로 꼽는다. 사전학습을 마친 특징 표현은 문맥에 의존하는 방법과 의존하지 않는 방법의 어느 방법도 있을 수 있다. 또 문맥에 의존하는 특징적인 표현은 단방향인 경우와 혹은 양방향일 경우가 있다. Word2Vec, GloVE와 같이 같이 문맥에 의존하지 않는 모델에서는, 어휘에 포함되는 각 단어마다 Word Embedding이라는 특징 표현을 생성한다. 따라서 bank라는 단어는 bank deposit 또는 river bank와 같은 특징으로 표현되며, 문맥에 의존하는 모델에서는 문장에 포함되는 다른 단어를 바탕으로 각 단어의 특징을 표현 생성한다. BERT는 문맥에 의존하는 특징적인 표현의 전학습을 실시하는 대응을 바탕으로 구축되었다. 그러한 대응은 Semi-supervised Sequence Learning, Generatvie Pre-Training, ELMo, 및 ULMFit를 포함하며, 대응에 의한 모델은 모두 단방향 혹은 양방향이다. 각 단어는 단지 그 왼쪽 혹은 오른쪽에 존재하는 단어에 의해서만 문맥의 고려가 되는 것을 의미한다. BERT는 간단한 접근법을 사용한다. 입력에서 단어의 15%를 숨기고 양방향 transformer encoder를 통해 전체 시퀀스를 실행한 다음 마스크된 단어만 예측한다. 또한 큰 모델(12 layers, 24 layers transformer)를 큰 코퍼스(Wikipedia + BookCorpus)로 긴 시간을 들여(100만 갱신 스텝)훈련했다. 이것이 BERT이며, 이용은 사전학습과 전이학습의 2단계로 구분된다. 이밖에 BERT의 또 다른 중요한 측면은 많은 종류의 자연어 처리 태스크로 인해 매우 쉽게 채택될 수 있다. 논문 중에서 문장 수준 (SST-2 등), 문장쌍 수준(MultiNLI 등), 단어 수준(NER 등), 스팬 레벨 2(SQuAD 등)의 태스크에 대해서 거의 태스크 특유의 변경을 실시하는 일 없이 최첨단 결과를 얻을 수 있는 것을 나타내고 있다.[Article]

### GPT-1 | [Homepage](https://openai.com/blog/language-unsupervised/) | [Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | [GitHub](https://github.com/openai/finetune-transformer-lm)
Improving Language Understanding with Generative Pre-Training

### GPT-2 | [Homepage](https://openai.com/blog/better-language-models/) | [Follow-up Post](https://openai.com/blog/gpt-2-6-month-follow-up/) | [Final Post](https://www.openai.com/blog/gpt-2-1-5b-release/) | [Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) | [GitHub](https://github.com/openai/gpt-2)
Language Models are Unsupervised Multitask Learners

### GPT-3 | [OpenAI API](https://openai.com/blog/openai-api/) | [Paper](https://arxiv.org/pdf/2005.14165.pdf) | [GitHub](https://github.com/openai/gpt-3)
Language Models are Few-Shot Learners

## Lecture
### 딥러닝을 이용한 자연어 처리 (조경현 교수님) | [Homepage](https://www.edwith.org/deepnlp)

## Reference
- Word Embedding Review Blog Kor, https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/03/11/embedding/, 2018-12-03-Mon.
- Word2Vec Review Blog Kor, https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/03/30/word2vec/, 2018-12-03-Mon.
- GloVe Review Blog Kor, https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/04/09/glove/, 2018-12-03-Mon.
- fastText, 2018-12-03-Mon.
- fastText Review Blog Kor, 2018-12-03-Mon.
- NPLM Review Blog Kor, https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/03/29/NNLM/, 2020-10-08-Mon.
- SVD, PCA and LSA Review Blog Kor, https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/04/06/pcasvdlsa/, 2020-10-08-Mon.
- PCA Review Blog Kor, https://ratsgo.github.io/machine%20learning/2017/04/24/PCA/, 2020-10-08-Mon.
- GPT-3 Article KR, https://www.aitimes.kr/news/articleView.html?idxno=17370, 2020-09-29-Tue.
- GPT-1 Review Kor, https://www.quantumdl.com/entry/12%EC%A3%BC%EC%B0%A81-Improving-Language-Understanding-by-Generative-Pre-Training, 2020-10-08-Thu.
- What is 'BERT'? Article KR, http://www.aitimes.kr/news/articleView.html?idxno=13117, 2020-11-30-Mon.
- GPT-3 License Article KR, http://www.aitimes.kr/news/articleView.html?idxno=17893, 2020-11-30-Mon.
- NLP Lecture KR, https://www.edwith.org/deepnlp, 2020-11-30-Mon.
