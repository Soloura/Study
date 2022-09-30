# Multimodal Learning | [Wiki](https://en.wikipedia.org/wiki/Multimodal_learning)

Information in the real world usually comes as different modalities. For example, images are usually associated with tags and text explanations; text contains images to more clearly express the main idea of the article. Different modalities are characterized by different statistical properties. For instance, images are
usually represented as pixel intensities or outputs of feature extractors, while texts are represented as discrete word count vectors. Due to the distinct statistical properties of different information resources, it is important to discover the relationship between different modalities. Multimodal learning is a good model to represent the joint representations of different modalities. The multimodal learning model is also capable of supplying a missing modality based on observed ones. The multimodal learning model combines two deep Boltzmann machines, each corresponding to one modalilty. An addtional hidden layer is placed on top of the two Boltzmann Machines to produce the joint representation.

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
