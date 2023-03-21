# Natural Language Processing

## Recurrent Neural Network [CS230: Deep Learning](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks)

Recurrent neural networks, also known as RNNs, are a class of neural networks that allow previous outputs to be used as inputs while having hidden states.

The pros and cons of a typical RNN architecture:
- Advantages:
  - Possibility of processing input of any length
  - Model size not increasing with size of input
  - Computation takes into account historical information
  - Weights are shared across time
- Drawbacks:
  - Computation being slow
  - Difficulty of accessing information from a long time ago
  - Cannot consider any future input for the current state

### Loss function

In the case of a recurrent neural network, the loss function L of all time steps is defined based on the loss at every time step as follows:

L(y_hat, y) = sum(L(y_hat^t, y^t))

### Backpropagation through time

Backpropagation is done at each point in time. At timestep T, the derivative of the loss L with respect to weight matrix W is experssed as follows:

dL^T / dW = sum(dL^T/dW)|_t

### GNU/LSTM

Gated Recurrent Unit(GRU) and Long Short-Term Memory units (LSTM) deal with the vanishing gradient problem encounterd by tranditional RNNs, with LSTM being a generalization of GRU. 

## Word Embedding | [Blog (KR)](https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/03/11/embedding/)
단어를 벡터로 바꾸는 방법론이다.

### Word2Vec

### Efficient Estimation of Word Representations in Vector Space | [arXiv 2013](https://arxiv.org/pdf/1301.3781.pdf)

### Distributed Representations of Words and Phrases and their Compositionality | [NIPS 2013](https://proceedings.neurips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf)

2013년 구글에서 개발한 워드 임베딩 방법론으로, 단어를 벡터로 바꾼다. 단어를 벡터로 바꾸고 문맥적 의미를 보존하여 단어들 간의 거리를 통해 비슷한 의미라 유추할 수 있다.

주변에 있는 단어들을 가지고 중심 단어를 유추하는 Continuous Bag of Words (CBOW) 방식과 중심에 있는 단어로 주변 단어를 예측하는 Skip-Gram 방식이 있다. 비슷한 위치에 등장하는 단어들은 그 의미도 유사할 것이라는 전제를 통해 distribution hypothesis에 근거한 방법론이다. 

### [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/) | [EMNLP, 2014](https://aclanthology.org/D14-1162.pdf)

2014년 스탠포드에서 개발한 워드 임베딩 방법론으로, 단어 동시 등장 여부를 보존하는 방식이다. GloVe로 임베딩된 단어 벡터끼리의 내적은 동시 등장확률의 로그 값과 같다.

Word2Vec이 임베딩된 두 단어 벡터의 내적이 코사인 유사도라면 GloVe는 동시 등장 확률이다.

### [fastText](https://research.fb.com/blog/2016/08/fasttext/)

2016년 페이스북에서 개발한 워드 임베딩 방법론으로, 원래 단어를 Word2Vec에서 기존 단어를 subword의 벡터들로 표현한다는 점이 추가된 내용이다. fastText 또한 Word2Vec와 같이 단어들의 동시 등장 정보를 보존한다.

Word2Vec, GloVe, fastText 모두 동시 등장 정보를 이용하기 때문에 다른 의미를 가진 단어들도 분포가 같다면 코사인 유사도가 높게 나오는 한계가 존재한다.

### Bag or Tricks for Efficient Text Classification | [EACL, 2017](https://aclanthology.org/E17-2068.pdf) | [arXiv](https://arxiv.org/abs/1607.01759) 

### Enriching Word Vectors with Subword Information | [TACL, 2017](https://aclanthology.org/Q17-1010.pdf) | [arXiv](https://arxiv.org/abs/1607.04606)

### Attention | *Neural Machine Translatin by Jointly Learning to Align and Translate* | [arXiv](https://arxiv.org/abs/1409.0473)

2015년 ICLR에 게재된 논문으로, Attention mechanism을 이용해 중요한 부분만 집중하게 하여 machine translation하는 내용이다.

### Transformer | [Wiki](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)) | [NIPS 2017](https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)

A transformer is a deep learning model that adopts the mechanism of self-attention, differentially weighting the significance of each part of the input data. It is used primarily in the fields of natural language processing (NLP) and computer vision (CV).

Like recurrent neural networks (RNNs), transformers are designed to process sequential input data, such as natural language, with applications towards tasks such as translation and text summarization. However, unlike RNNs, transformers process the entire input all at once. The attention mechanism provides context for any position in the input sequence. For example, if the input data is a natural language sentence, the transformer does not have to process one word at a time. This allows for more parallization than RNNs and therefore reduces training times.

Transformers were introduced in 2017 by a team at Google Brain and are increasingly the model of choice for NLP problems, replacing RNN models such as long short-term memory (LSTM). The additional training parallelization allows training on larger datasets. This led to the development of pretrained systems such as BERT (Bidirectional Encoder Representations from Transformers) and GPT (Generative Pre-trained Transformer), which were trained with large language datasets, such as the Wikipedia Corpus and Common Crawl, and can be fine-tuned for specific tasks.

### NPLN(Neural Probabilistic Language Model)

### SVD(Singular Value Decomposion)

### PCA(Principal Component Analysis)

### LSA(Latent Sematic Analysis)

### Embeddings from Language Model(ELMo) | [Paper (Homepage)](https://www.aclweb.org/anthology/N18-1202/) | [Paper (arXiv)](https://arxiv.org/pdf/1802.05365.pdf)

### Doc2Vec

---

### Bidirectional Encoder Representations from Transformers(BERT)

2018년 11월에 Google이 공개한 언어모델 BERT는 일부 성능 평가에서 인간보다 더 높은 정확도를 보이며 2018년 말, 상위 딥러닝 모델이다. 또한 BERT는 언어표현 사전학습의 새로운 방법으로 그 의미는 큰 텍스트 코퍼스(Wikipedia와 같은)를 이용하여 범용목적의 언어 이해(Language Understanding) 모델을 훈련시키는 것과 그 모델에 관심 있는 실제의 자연 언어 처리 태스트(질문, 응답 등)에 적용하는 것이다. 특히 BERT는 종래보다 우수한 성능을 발휘한다. BERT는 자연어 처리 태스크를 교육 없이 양방향으로 이용해 훈련되고 있다는 것을 의미한다. 이것은 Web 상에서 막대한 양의 보통 텍스트 데이터가 여러 언어로 이용 가능하기 때문에 중요한 특징으로 꼽는다. 사전학습을 마친 특징 표현은 문맥에 의존하는 방법과 의존하지 않는 방법의 어느 방법도 있을 수 있다. 또 문맥에 의존하는 특징적인 표현은 단방향인 경우와 혹은 양방향일 경우가 있다. Word2Vec, GloVE와 같이 같이 문맥에 의존하지 않는 모델에서는, 어휘에 포함되는 각 단어마다 Word Embedding이라는 특징 표현을 생성한다. 따라서 bank라는 단어는 bank deposit 또는 river bank와 같은 특징으로 표현되며, 문맥에 의존하는 모델에서는 문장에 포함되는 다른 단어를 바탕으로 각 단어의 특징을 표현 생성한다. BERT는 문맥에 의존하는 특징적인 표현의 전학습을 실시하는 대응을 바탕으로 구축되었다. 그러한 대응은 Semi-supervised Sequence Learning, Generatvie Pre-Training, ELMo, 및 ULMFit를 포함하며, 대응에 의한 모델은 모두 단방향 혹은 양방향이다. 각 단어는 단지 그 왼쪽 혹은 오른쪽에 존재하는 단어에 의해서만 문맥의 고려가 되는 것을 의미한다. BERT는 간단한 접근법을 사용한다. 입력에서 단어의 15%를 숨기고 양방향 transformer encoder를 통해 전체 시퀀스를 실행한 다음 마스크된 단어만 예측한다. 또한 큰 모델(12 layers, 24 layers transformer)를 큰 코퍼스(Wikipedia + BookCorpus)로 긴 시간을 들여(100만 갱신 스텝)훈련했다. 이것이 BERT이며, 이용은 사전학습과 전이학습의 2단계로 구분된다. 이밖에 BERT의 또 다른 중요한 측면은 많은 종류의 자연어 처리 태스크로 인해 매우 쉽게 채택될 수 있다. 논문 중에서 문장 수준 (SST-2 등), 문장쌍 수준(MultiNLI 등), 단어 수준(NER 등), 스팬 레벨 2(SQuAD 등)의 태스크에 대해서 거의 태스크 특유의 변경을 실시하는 일 없이 최첨단 결과를 얻을 수 있는 것을 나타내고 있다.[Article]

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

## Speech Recognition

### Whisper - A model that can convert audio into text | [OpenAI](https://openai.com/blog/whisper/) | [GitHub](https://github.com/openai/whisper) | [Paper](https://cdn.openai.com/papers/whisper.pdf)

Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multi-task model that can perform multilingual speech recognition as well as speech translation and language indentification.

---

## Lecture

### [딥러닝을 이용한 자연어 처리 (조경현 교수님)](https://www.edwith.org/deepnlp)

---

### Reference
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
- ELMo Paper, https://www.aclweb.org/anthology/N18-1202/ 2020-12-01-Tue.
- ELMo Blog KR, https://wikidocs.net/33930, 2020-12-01-Tue.
- Attentions Mechanism Blog KR, https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/10/06/attention/, 2021-10-15-Fri.
- Linguistic Model ScienceDirect, https://www.sciencedirect.com/topics/computer-science/linguistic-model, 2022-09-30-Fri.
- OpenAI ChatGPT, https://openai.com/blog/chatgpt/, 2022-12-09-Fri.
- OpenAI ChatGPT Try, https://chat.openai.com/auth/login, 2022-12-09-Fri.
- OpenAI InstructGPT, https://openai.com/blog/instruction-following/, 2022-12-09-Fri.
- OpenAI InstructGPT arXiv, https://arxiv.org/abs/2203.02155, 2022-12-09-Fri.
- Whisper, https://openai.com/blog/whisper/, 2022-12-10-Sat.
- Whisper GitHub, https://github.com/openai/whisper, 2022-12-10-Sat.
- Whisper Paper, chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://cdn.openai.com/papers/whisper.pdf, 2022-12-10-Sat.
- Recurrent Nerual Network CS230, https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks, 2023-03-01-Wed.
- GPT-4 OpenAI, https://openai.com/product/gpt-4, 2023-03-16-Thu.
- Transformer Wiki, https://en.wikipedia.org/wiki/Transformer_(machine_learning_model), 2023-03-21-Tue.
- Attention is All you Need, https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html, 2023-03-21-Tue.
