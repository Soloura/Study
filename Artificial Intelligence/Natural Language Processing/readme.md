# Natural Language Processing (NLP) | [IBM](https://www.ibm.com/topics/natural-language-processing)

Natural Language Processing (NLP) refers to the branch of computer science - and more specifically, the branch of artificial intelligence or AI - concerned with giving computers the ability to understand text and spoken words in much the same way human beings can.

NLP combines computational linguistics - rule-based modeling of human language - with statistical, machine learning, and deep learning models. Together, these technologies enable computers to process human language in the form of text or voice data and to 'understand' its full meaning, complete with the speaker or writer's intent and sentiment.

NLP drives computer programs that translate text from one language to another, respond to spoken commands, and summarize large volumes of text rapidly - even in real time. There's a good chance you've interacted with NLP in the form of voice-operated GPS systems, digital assistants, speech-to-text dictation software, customer service chatbots, and other consumer conveniences. But NLP also plays a growing role in enterprise solutions that help streamline business operations, increase employee productivity, and simplify mission-critical business processes.

## Recurrent Neural Network | [CS230: Deep Learning](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks)

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

### GRU/LSTM

Gated Recurrent Unit(GRU) and Long Short-Term Memory units (LSTM) deal with the vanishing gradient problem encounterd by tranditional RNNs, with LSTM being a generalization of GRU.


### GRU - Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation | [EMNLP 2014](https://aclanthology.org/D14-1179/)

Gated recurrent units (GRUs) are a gating mechanism in recurrent neural networks, introduced in 2014 by Cho et al. The GRU is like a long short-term memory (LSTM) with a forget gate, but has fewer parameters than LSTM, as it lacks an output gate. GRU's performance on certain tasks of polyphonic music modeling, speech signal modeling and natural language processing was found to be similar to that of LSTM. GRUs shown that gating is indeed helpful in general and Bengio's team concluding that no concrete conclusion on which of the two gating units was better.

Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling | [NIPS 2014](https://nyuscholars.nyu.edu/en/publications/empirical-evaluation-of-gated-recurrent-neural-networks-on-sequen) | [arXiv 2014](https://arxiv.org/abs/1412.3555)

Are GRU cells more specific and LSTM cells more sensitive in motive classification of text?", Frontiers in Artificial Intelligence | [Frontiers in AI 2020](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7861254/)

---

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

### Attention - Neural Machine Translatin by Jointly Learning to Align and Translate | [arXiv](https://arxiv.org/abs/1409.0473) | [Wiki](https://en.wikipedia.org/wiki/Attention_(machine_learning))

2015년 ICLR에 게재된 논문으로, Attention mechanism을 이용해 중요한 부분만 집중하게 하여 machine translation하는 내용이다.

An attention is a technique that is meant to mimic cognitive attention. The effect enchances some parts of the input data while diminishing other parts - the motivation being that the network should devote more focus to the small, but important, parts of the data. Learning which part of the data is more important than another depends on the context, and this is trained by gradient descent.

Attention-like mechanisms were introduced in the 1990s under name like multiplicative modules, sigma pi units, and hyper-networks. Its flexibility comes from its role as "soft weights: that can change during runtime, in contrast to standard weights that must remain fixed at runtime. Uses of attention include memory in neural Turing machines, reasoning tasks in differentiable neural computers, language processing in transformers, and LSTMs, and multi-sensory data processing (sound, images, video, and text) in perceivers. Listed in the Variants section below are the many schemes to implement the soft-weight mechanisms.

### Transformer - Attention is All you Need | [NIPS 2017](https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) | [Wiki](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model))

A transformer is a deep learning model that adopts the mechanism of self-attention, differentially weighting the significance of each part of the input data. It is used primarily in the fields of natural language processing (NLP) and computer vision (CV).

Like recurrent neural networks (RNNs), transformers are designed to process sequential input data, such as natural language, with applications towards tasks such as translation and text summarization. However, unlike RNNs, transformers process the entire input all at once. The attention mechanism provides context for any position in the input sequence. For example, if the input data is a natural language sentence, the transformer does not have to process one word at a time. This allows for more parallization than RNNs and therefore reduces training times.

Transformers were introduced in 2017 by a team at Google Brain and are increasingly the model of choice for NLP problems, replacing RNN models such as long short-term memory (LSTM). The additional training parallelization allows training on larger datasets. This led to the development of pretrained systems such as BERT (Bidirectional Encoder Representations from Transformers) and GPT (Generative Pre-trained Transformer), which were trained with large language datasets, such as the Wikipedia Corpus and Common Crawl, and can be fine-tuned for specific tasks.

### NPLN (Neural Probabilistic Language Model)

### SVD (Singular Value Decomposion)

### PCA (Principal Component Analysis)

### LSA (Latent Sematic Analysis)

### Embeddings from Language Model(ELMo) | [Paper (Homepage)](https://www.aclweb.org/anthology/N18-1202/) | [Paper (arXiv)](https://arxiv.org/pdf/1802.05365.pdf)

### Doc2Vec

---

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

## Speech Recognition

### Whisper - A model that can convert audio into text | [OpenAI](https://openai.com/blog/whisper/) | [GitHub](https://github.com/openai/whisper) | [Paper](https://cdn.openai.com/papers/whisper.pdf)

Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multi-task model that can perform multilingual speech recognition as well as speech translation and language indentification.

## [Text Generation](https://huggingface.co/tasks/text-generation)

Generating text is the task of producing new text. These models can, for example, fill in incomplete text of paraphrase.

---

## Lecture

- [CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/)
- [딥러닝을 이용한 자연어 처리 (조경현 교수님)](https://www.edwith.org/deepnlp)
- [Hugging Face Course](https://huggingface.co/learn/nlp-course/chapter1/1) [KR](https://wikidocs.net/book/8056)

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
- Attention Wiki, https://en.wikipedia.org/wiki/Attention_(machine_learning), 2023-03-21-Tue.
- Transformer Wiki, https://en.wikipedia.org/wiki/Transformer_(machine_learning_model), 2023-03-21-Tue.
- Attention is All you Need, https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html, 2023-03-21-Tue.
- BERT Article KR, https://www.aitimes.kr/news/articleView.html?idxno=13117, 2023-03-21-Tue.
- BERT arXiv, https://arxiv.org/pdf/1810.04805.pdf, 2023-03-21-Tue.
- BERT GitHub, https://github.com/google-research/bert, 2023-03-21-Tue.
- BERT Wiki, https://en.wikipedia.org/wiki/BERT_(language_model), 2023-03-21-Tue.
- Natural Language Processing IBM, https://www.ibm.com/topics/natural-language-processing, 2023-03-21-Tue.
- GRU Wiki, https://en.wikipedia.org/wiki/Gated_recurrent_unit, 2023-03-21-Tue.
- GRU EMNLP 2014, https://aclanthology.org/D14-1179/, 2023-03-21-Tue.
- Text Generation Hugging Face, https://huggingface.co/tasks/text-generation, 2023-10-18-Wed.
- Hugging Face Course, https://huggingface.co/learn/nlp-course/chapter1/1, 2023-10-18-Wed.
- Hugging Face Course KR, https://wikidocs.net/book/8056, 2023-10-18-Wed.
