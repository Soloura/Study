# Multimodal Learning | [Wiki](https://en.wikipedia.org/wiki/Multimodal_learning)

Information in the real world usually comes as different modalities. For example, images are usually associated with tags and text explanations; text contains images to more clearly express the main idea of the article. Different modalities are characterized by different statistical properties. For instance, images are
usually represented as pixel intensities or outputs of feature extractors, while texts are represented as discrete word count vectors. Due to the distinct statistical properties of different information resources, it is important to discover the relationship between different modalities. Multimodal learning is a good model to represent the joint representations of different modalities. The multimodal learning model is also capable of supplying a missing modality based on observed ones. The multimodal learning model combines two deep Boltzmann machines, each corresponding to one modalilty. An addtional hidden layer is placed on top of the two Boltzmann Machines to produce the joint representation.

---

## Overview | [Blog (KR)](https://seunghan96.github.io/mult/study-(multi)Multimodal-Learning%EC%86%8C%EA%B0%9C/)

Multimodal learning은 다양한 형태 (modality) 데이터를 사용하여 모델을 학습한다.: vision, text, speech, touch, smell, and meta data

변수들의 차원이 다르다.

Expression:

Single-modal: y = f(X_n^p)

Multi-modal: y = f(X_term^doc, X_(x, y)^color, X_time^voice, X_time^sensor)

Key Point:

어떻게 특징 차원이 다른 데이터를 동시에 잘 학습할 수 있을까? - 각각의 데이터의 특성을 잘 통합하는 것에 있다.

Multimodal learning은 여러 source의 데이터를 통합하는 방식에 따라 구분할 수 있다.

1. 다른 특성의 데이터를 embedding하여 특성이 같은 데이터로 추출한다. ex) Deep CCA (Deep Canonical Correlation Analysis). (u, v) =argmax corr (uTX, vTY)

2. 각기 다른 Model의 예측 값을 통합한다(Co-training, Ensemble). 각각의 모델은 다른 가중치를 가지고 통합된다. P(Y) = sum{text, speech, vision}gamma*P(Y) (gamma = weight)

3. 각각의 데이터는 각자의 neural network를 통해 학습된 뒤, 거기서 추출된 embedding vector를 (선형) 결합한다.
ex1) Multimodal CNN (m-CNN): image와 text의 matching 정도 관계 파악 - image/text 특징을 concatenate하는 nn, concatenate된 vector로써 최종 예측을 하는 nn,
ex2) Multimodal RNN (m-RNN): image와 관련된 text를 생성 - 시계열 특성을 파악하는 nn, image/text의 특징을 concatenate하는 nn

Challenges:

1. Representation: multimodal data를 어떻게 잘 highly correlated할 것인지?
method1) joint representation: 두 data가 합쳐진 뒤 하나의 representation.
method2) coordinated representation: 두 data가 각각 축약된 뒤, 이들을 서로 concatemate - ex) Deep Data

2. Translation: entity를 다른 modality의 entity로 변환/생성

3. Alignment: 서로 다른 modality의 데이터의 관계를 파악

4. Fusion: 서로 다른 modality의 데이터를 잘 결합하여 예측을 수행

5. Co-learning: knowledge가 부족한 특정 modality의 데이터를 knowledge가 풍부한 다른 modality의 데이터를 사용하여 보완

## Survey | [Blog (KR)](https://techy8855.tistory.com/20)

1. Representation: multiple modalities를 어떻게 잘 결합할 것인가? - Joint method, Combinded method, nn의 vector representation을 어떻게 할 것인가.
2. Translation: 하나의 modality가 다른 modality로 옮겨져 갈 때, 이 관계를 학습한다. Uni-modality로는 한국-영어 번역 모델이 있고, multi-modality의 경우, 문장을 쓰면 그 문장이 이미지화되는 모델이다.
3. Alignment: multi modality 간의 직접적인 관계를 학습하는 것으로, multimodal pair (x, y, z)를 align(pairing)하여 조합 자체를 학습한다.
4. Fusion: multi-modality에서 나온 각각의 정보를 잘 조합해서 최종 예측에 사용한다.

---

## Papers

### Using Both Demonstrations and Language Instructions to Efficiently Learn Robotics Tasks | ICLR 2023

Train a sinle multi-task policy on a few hundred challenging robotic pick-and-place tasks and propose DeL-TaCo (Joint Demo-Lnaguage Task Conditioning), a method for conditioning a robotic policy on task embeddings comprised of two components: a visual demonstraction and a language instruction.

DeL-TaCo:
1. Substantially decreases the teacher effort needed to specify a new task
2. achieves better generalization performance on novel objects and instructions over previous task-conditioning methods

- Training: Train a single multi-task policy on hundreds of tasks.
- Testing: One-shot generalization to ~100 new tasks (new objects, colors, shapes).

---

### Related Works

### Multi-task Learning

- condition multi-task policies one-hot vectors
  - Bridge data: Boosting generalization of robotic skills with cross-domain datasets (arXiv 2021)
  - Mt-opt: Continuous multi-task robotic reinforcement learning at scale (arXiv 2021)
  - Don’t start from scratch: Leveraging prior data to automate robotic reinforcement learning (CoRL 2022)
  - Conservative data sharing for multi-task offline reinforcement learning (NIPS 2021)
- Embedding spaces taht are shaped with pretrained language models
  - Hierarchical goal-conditioned policies
    - Demonstration-bootstrapped autonomous practicing via multi-task reinforcement learning (arXiv 2022)
  - Probabilistic modeling techniques
    - Multi-task reinforcement learning: a hierarchical bayesian approach (ICML 2007)
  - distillation and transfer learning
    - Actor-mimic: Deep multitask and transfer reinforcement learning (arXiv 2015)
    - Distral: Robust multitask reinforcement learning (arXiv 2017)
    - Knowledge transfer in multi-task deep reinforcement learning for continuous control (arXiv 2020)
    - Policy distillation (arXiv 2015)
  - Data sharing
    - Impala: Scalable distributed deep-rl with importance weighted actor-learner architectures (ICML 2018)
    - Multi-task deep reinforcement learning with popart (AAAI 2019)
  - Gradient-based techniques
    - Gradient surgery for multi-task learning (arXiv 2020)
  - Policy modularization
    - Modular multitask reinforcement learning with policy sketches (ICML 2017)
    - Learning modular neural network policies for multi-task and multi-robot transfer (ICRA 2017)
  - Task modularization
    - Multi-task reinforcement learning with soft modularization (arXiv 2020)

### Learning with Language and Demonstratins

- Conditioning Multitask Policies on Language or Demonstrations
  - BC-Z
    - BC-z: Zero-shot task generalization with robotic imitation learning (CoRL 2021)
  - either the instruction or demonstration embeddings
    - Language conditioned imitation learning over unstructured data (arXiv 2021)
    - Calvin: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks (arXiv 2021)
  - both demonstrations and language to learn associations between demenstration embeddings and language-conditioned latent plans
    - What matters in language conditioned imitation learning (arXiv 2022)
  - learn a policy that maps natural language verbs and initial observations to full trajectories by training a video classifier on a large dataset of annotated human vides
    - Concept2robot: Learning manipulation concepts from instructions and human demonstrations (RSS 2020)
- Pretrained Multi-modal Models for Multitask Policies
  - pretrained vision-language models to learn richer vision features for downstream policies.
    - CLIPort
      - Cliport: What and where pathways for robotic manipulation (CoRL 2021)
    - CLIP
      - Learning transferable visual models from natural language supervision (arXiv 2021)
    - Transporter-based
      - Transporter networks: Rearranging the visual world for robotic manipulation (CoRL 2020)
    - PerAct
      - A multi-task transformer for robotic manipulation (CoRL 2022)
  - ZeST
    - Can foundation models perform zero-shot task specification for robot manipulation? (LDCC 2022)
  - Socratic Models
    - Socratic models: Composing zero-shot multimodal reasoning with language (arXiv 2022)
  - R3M
    - R3m: A universal visual representation for robot manipulation (arXiv 2022)
  - ResNet
    - Deep residual learning for image recognition (arXiv 2015)
  - Ego4D
    - Ego4d: Around the world in 3,000 hours of egocentric video (arXiv 2021)

### Other Applications of Language for Robotics

- Language-shaped state representations
  - Meta World multitask benchmark
    - Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning (CoRL 2019)
- Hierarchical Learning with Language
  - shaping high-level plan vectors
    - What matters in language conditioned imitation learning (arXiv 2022)
  - skill representation
    - Lisa: Learning interpretable skill abstractions from language (arXiv 2022)
  - low-level policy to output the action
    - Lila: Language-informed latent actions (CoRL 2021)
- Language for Rewards and Planning
  - Reward shaping in RL
    - Learning language-conditioned robot behavior from offline data and crowd-sourced annotation (arXiv 2021)
    - Using natural language for reward shaping in reinforcement learning (arXiv 2019)
    - Pixl2r: Guiding reinforcement learning using natural language by mapping pixels to rewards (arXiv 2020)
  - long-horizon tasks
    - Inner monologue: Embodied reasoning through planning with language models (arXiv 2022)
    - Do as I can and not as I say: Grounding language in robotic affordances (arXiv 2022)
    - Open-vocabulary queryable scene representations for real world planning (arXiv 2022)

---

## Models

### A Generalist Agent | [DeepMind](https://www.deepmind.com/publications/a-generalist-agent)
The agent, Gato, works as a multi-modal, multi-task, multi-embodiment generalist policy. The same network with the same weights can play Atari, caption images, chat, stack blocks with a real robot arm and much more, deciding based on its context whether to ouput text, joint torques, button presses, or other tokens.

### [DALL-E](https://openai.com/blog/dall-e/) by OpenAI | [GitHub](https://github.com/openai/DALL-E)

DALL-E is a 12-billion parameter version of GTP-3 trained to generate images from text descriptions, using a dataset of text-image pairs.

### [DALL-E 2](https://openai.com/dall-e-2/) by OpenAI | [Paper](https://cdn.openai.com/papers/dall-e-2.pdf) | [GitHub](https://github.com/lucidrains/DALLE2-pytorch)

DALL-E 2 is a new AI system that can create realistic image and art from a description in natural language. DALL-E 2 can create original, realistic images and art from a text description. It can combine concepts, attributes, and styles. DALL-E 2 can expand images beyond what's in the original canvas, creating expansive new compositions. DALL-E 2 can make realistic edits to existing images from a natural language caption. It can add and remove elements while taking shadows, reflections, and textures into account. DALL-E 2 can take an image and create different variations of it inspired by the original. DALL-E 2 has learned the relationship between iamges and the texture used to describe them. It uses a process called "diffusion", which starts with a parttern of random dots and gradually alters that pattern towards an image when it recognizes specific aspects of that image.

### [Imagen](https://imagen.research.google/) by Google Research, Brain Team | [GitHub](https://github.com/lucidrains/imagen-pytorch)

Unprecedented photorealism x deep level of language understanding.

Imagen is an AI system that creates photorealistic images from input text. Visualization of Imagen. Imagen uses a large frozen T5-XXL encoder to encode the input text into embeddings. A conditional diffusion model maps the text embedding into a 64x64 image. Imagen further utilizes text-conditional super-resolution diffusion modesl to upsample the image 64x64 -> 256x256 and -> 1024x1024.

Large Pretrained Language Model x Cascaded Diffusion Model: deep textual understanding -> photorealistic generation.

Imagen research highlights
1. We show that large pretrained frozen text encdoers are very effective for the text-to-image task.
2. We show that scaling the pretrained text encoder size is more important than scaling the diffusion model size.
3. We introduce a new thresholding diffusion sampler, which enables the use of very large classifier-free guidance weights.
4. We introduce a new Efficient U-Net architecture, which is more compute efficient, more memory efficient, and converges faster.
5. On COCO, we achieve a new state-of-the-art COCO FID of 7.27; and human raters find Imagen samples to be on-par with reference images in terms of image-text alignment.

DrawBench: new comprehensive challenging benchmark
1. Side-by-side human evaluation
2. Systematically test for: compositionality, cardinality, spatial relations, long-form text, rare words, and challenging prompts.
3. Human raters strongly prefer Imagen over other methods, in both image-text aligment and image fidelity.

Imagen: imagine, illustrate, inspire.

---

### Reference
- Multimodal learning Wiki, https://en.wikipedia.org/wiki/Multimodal_learning, 2021-12-13-Mon.
- A Generalist Agent, https://www.deepmind.com/publications/a-generalist-agent, 2022-06-23-Thu.
- DALL-E, https://openai.com/blog/dall-e/, 2022-09-30-Fri.
- DALL-E arXiv, https://arxiv.org/abs/2102.12092, 2022-09-30-Fri.
- DALL-E GitHub, https://github.com/openai/DALL-E, 2022-09-30-Fri.
- DALL-E Blog KR, https://littlefoxdiary.tistory.com/74?category=847374, 2022-09-30-Fri.
- DALL-E 2, https://openai.com/dall-e-2/, 2022-09-30-Fri.
- DALL-E 2 Paper, https://cdn.openai.com/papers/dall-e-2.pdf, 2022-09-30-Fri.
- DALL-E 2 GitHub, https://github.com/lucidrains/DALLE2-pytorch, 2022-09-30-Fri.
- Imagen, https://imagen.research.google/, 2022-09-30-Fri.
- Imagen GitHub, https://github.com/lucidrains/imagen-pytorch, 2022-09-30-Fri.
- Imagen arXiv, https://arxiv.org/abs/2205.11487, 2022-09-30-Fri.
- Multimodal Learning Overview Blog KR, https://seunghan96.github.io/mult/study-(multi)Multimodal-Learning%EC%86%8C%EA%B0%9C/, 2022-10-23-Sun.
- Multimodal Learning Survey Blog KR, https://techy8855.tistory.com/20, 2022-10-24-Mon.
