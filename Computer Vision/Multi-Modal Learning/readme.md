# Multimodal Learning | [Wiki](https://en.wikipedia.org/wiki/Multimodal_learning)

Information in the real world usually comes as different modalities. For example, images are usually associated with tags and text explanations; text contains images to more clearly express the main idea of the article. Different modalities are characterized by different statistical properties. For instance, images are
usually represented as pixel intensities or outputs of feature extractors, while texts are represented as discrete word count vectors. Due to the distinct statistical properties of different information resources, it is important to discover the relationship between different modalities. Multimodal learning is a good model to represent the joint representations of different modalities. The multimodal learning model is also capable of supplying a missing modality based on observed ones. The multimodal learning model combines two deep Boltzmann machines, each corresponding to one modalilty. An addtional hidden layer is placed on top of the two Boltzmann Machines to produce the joint representation.

----------

#### Reference
- Multimodal learning Wiki, https://en.wikipedia.org/wiki/Multimodal_learning, 2021-12-13-Mon.
