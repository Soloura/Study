# [Transformer](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)) | [NVIDIA](https://blogs.nvidia.com/blog/what-is-a-transformer-model/) | [AWS](https://aws.amazon.com/what-is/transformers-in-artificial-intelligence/)

A transformer is a deep learning architecture developed by Google and based on the multi-head attention mechanism, proposed in a 2017 paper "Attention Is All You Need". It has no recurrent units, and thus requires less training time than previous recurrent neural architectures, such as long short-term memory (LSTM), and its later variation has been prevalently adopted for training large language models (LLM) on large (language) datasets, such as the Wikipedia corpus and Common Crawl. Text is converted to numerical representations called tokens, and each token is converted into a vector via looking up from a word embedding table. At each layer, each token is then contextualized within the scope of the context window with other (unmasked) tokens via a parallel multi-head attention mechanism allowing the signal for key tokens to be amplified and less important tokens to be diminished. The transformer paper, published in 2017, is based on the softmax-based attention mechanism proposed by Bahdanau et. al. in 2014 for machine translation, and the Fast Weight Controller, similar to a transformer, proposed in 1992.

This architecture is now used not only in natural language processing and computer vision, but also in audio and multi-modal processing. It has also led to the development of pre-trained systems, such as generative pre-trained transformers (GPTs) and BERT (Bidirectional Encoder Representations from Transformers).

## Incoder-Decoder

### T5

### BART

### M2M-100

### BigBird

## Incoder

### BERT

### DistilBERT

### RoBERTa

### XLM

### XLM-RoBERTa

### ALBERT

### ELECTRA

### DeBERTa

## Decoder

### GPT

### GPT-2

### CTRL

### GPT-3

##3 GPT-Neo/GPT-J-6B

---

### Reference
- Transformer Nvidia, https://blogs.nvidia.com/blog/what-is-a-transformer-model/, 2024-02-06-Tue.
- Transformer AWS, https://aws.amazon.com/what-is/transformers-in-artificial-intelligence/, 2024-02-06-Tue.
- Transformer 1 KR Nvidia, https://blogs.nvidia.co.kr/2022/04/01/what-is-a-transformer-model/, 2024-03-12-Tue.
- Transformer 2 KR Nvidia, https://blogs.nvidia.co.kr/2022/04/01/what-is-a-transformer-model-2/, 2024-03-12-Tue.
- Transformer Models Blog KR, https://velog.io/@jx7789/%EB%8B%A4%EC%96%91%ED%95%9C-%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B8-%EB%AA%A8%EB%8D%B8%EB%93%A4-l3z5ap4p, 2024-03-12-Tue.
- Transformer Wiki, https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture), 2024-03-12-Tue.
