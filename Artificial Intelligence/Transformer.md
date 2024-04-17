# [Transformer](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture))

A transformer is a deep learning architecture developed by Google and based on the multi-head attention mechanism, proposed in a 2017 paper "Attention Is All You Need". It has no recurrent units, and thus requires less training time than previous recurrent neural architectures, such as long short-term memory (LSTM), and its later variation has been prevalently adopted for training large language models (LLM) on large (language) datasets, such as the Wikipedia corpus and Common Crawl. Text is converted to numerical representations called tokens, and each token is converted into a vector via looking up from a word embedding table. At each layer, each token is then contextualized within the scope of the context window with other (unmasked) tokens via a parallel multi-head attention mechanism allowing the signal for key tokens to be amplified and less important tokens to be diminished. The transformer paper, published in 2017, is based on the softmax-based attention mechanism proposed by Bahdanau et. al. in 2014 for machine translation, and the Fast Weight Controller, similar to a transformer, proposed in 1992.

This architecture is now used not only in natural language processing and computer vision, but also in audio and multi-modal processing. It has also led to the development of pre-trained systems, such as generative pre-trained transformers (GPTs) and BERT (Bidirectional Encoder Representations from Transformers).

[NVIDIA](https://blogs.nvidia.com/blog/what-is-a-transformer-model/) | [AWS](https://aws.amazon.com/what-is/transformers-in-artificial-intelligence/)

## :books: Natural Language Processing

* Incoder-Decoder
  * T5
  * BART
  * M2M-100
  * BigBird
* Incoder
  * BERT
  * DistilBERT
  * RoBERTa
  * XLM
  * XLM-RoBERTa
  * ALBERT
  * ELECTRA
  * DeBERTa
* Decoder
  * GPT
  * GPT-2
  * CTRL
  * GPT-3
  * GPT-Neo/GPT-J-6B

## :books: Computer Vision

### :bookmark_tabs: An Image is Worth 16x16 Words Transformers for Image Recognition at Scale | [2021 ICLR](https://openreview.net/pdf?id=YicbFdNTTy) | [2021 arXiv](https://arxiv.org/abs/2010.11929)

While the Transformers architecture has become the de-facto standard for natural lanuage processing tasks, its applications to computer vision ramain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.

[Blog (KR)](https://kmhana.tistory.com/27) | [Blog (KR)](https://hipgyung.tistory.com/entry/%EC%89%BD%EA%B2%8C-%EC%9D%B4%ED%95%B4%ED%95%98%EB%8A%94-ViTVision-Transformer-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-An-Image-is-Worth-16x16-Words-Transformers-for-Image-Recognition-at-Scale)

### :bookmark_tabs: Visual Prompt Tuning | [2022 ECCV](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930696.pdf)

The current modus operandi in adapting pre-trained models involves updating all the backbone parameters, i.e., full fine-tuning. This paper introduces Visual Prompt Tuning (VPT) as an efficient and effective alternative to full fine-tuning for large-scale Transformer models in vision. Taking inspiration from recent advances in efficiently tuning large language models, VPT introduces only a small amount (less than 1% of model parameters) of trainable parameters in the input space while keeping the model backbone frozen. Via extensive experiments on a wide variety of downstream recognition tasks, we show that VPT achieves significant performance gains compared to other parameter efficient tuning protocols. Most improtantly, VPT even outperforms full fine-tuning in many cases across model capacities and training data scales, while reducing per-task storage cost.

### :bookmark_tabs: [Segment Anything](https://segment-anything.com/) | [2023 ICCV](https://openaccess.thecvf.com/content/ICCV2023/papers/Kirillov_Segment_Anything_ICCV_2023_paper.pdf) | [2023 arXiv](https://arxiv.org/abs/2304.02643)

The Segment Anything (SA) project introduces a new task, model, and dataset for image segmentation. Their efficient model, designed to be promptable, enabled the creation of the largest segmentation dataset to date, with over 1 billion masks on 11 million licensed and privacy-respecting images. SAM, the Segment Anything Model, and SA-1B dataset are released at segment-anything.com to encourage research into foundation models for computer vision, showing impressive zero-shot performance on various tasks.

---

### Reference
- Transformer Nvidia, https://blogs.nvidia.com/blog/what-is-a-transformer-model/, 2024-02-06-Tue.
- Transformer AWS, https://aws.amazon.com/what-is/transformers-in-artificial-intelligence/, 2024-02-06-Tue.
- Transformer 1 KR Nvidia, https://blogs.nvidia.co.kr/2022/04/01/what-is-a-transformer-model/, 2024-03-12-Tue.
- Transformer 2 KR Nvidia, https://blogs.nvidia.co.kr/2022/04/01/what-is-a-transformer-model-2/, 2024-03-12-Tue.
- Transformer Models Blog KR, https://velog.io/@jx7789/%EB%8B%A4%EC%96%91%ED%95%9C-%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B8-%EB%AA%A8%EB%8D%B8%EB%93%A4-l3z5ap4p, 2024-03-12-Tue.
- Transformer Wiki, https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture), 2024-03-12-Tue.
- Segment Anything, https://segment-anything.com/, 2024-04-03-Wed
- Segment Anything, https://openaccess.thecvf.com/content/ICCV2023/papers/Kirillov_Segment_Anything_ICCV_2023_paper.pdf, 2024-04-03-Wed.
- Visual Prompt Tuning, https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930696.pdf, 2024-04-03-Wed.
- An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale ICLR 2021, https://openreview.net/pdf?id=YicbFdNTTy, 2024-04-04-Thu.
- An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale Blog KR, https://kmhana.tistory.com/27, 2024-04-04-Thu.
- An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale Blog KR, https://hipgyung.tistory.com/entry/%EC%89%BD%EA%B2%8C-%EC%9D%B4%ED%95%B4%ED%95%98%EB%8A%94-ViTVision-Transformer-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-An-Image-is-Worth-16x16-Words-Transformers-for-Image-Recognition-at-Scale, 2024-04-04-Thu.
- An Image is Worth 16x16 Words Transformers for Image Recognition at Scale 2021 ICLR, https://openreview.net/pdf?id=YicbFdNTTy, 2024-04-17-Wed.
- An Image is Worth 16x16 Words Transformers for Image Recognition at Scale 2021 arXiv, https://arxiv.org/abs/2010.11929, 2024-04-17-Wed.
- ViT Blog KR, https://kmhana.tistory.com/27, 2024-04-17-Wed.
- Segment Anything GitHub, https://github.com/facebookresearch/segment-anything, 2024-04-17-Wed.
- Segment Anything 2023 arXiv, https://arxiv.org/abs/2304.02643, 2024-04-17-Wed.
