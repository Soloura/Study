# Few-Shot Learning | [Article](https://www.analyticsvidhya.com/blog/2021/05/an-introduction-to-few-shot-learning/)
`This page is moved from the "Computer Vision" page.`

Few-shot learning means making classification or regression based on a very small number of samples or even a zero number of  sample. It is different with other normal deep learning methods. The goal of few-shot learning is not to let the model recognize the images in the training set and then generalize to the test set. Instead, the goal is to learn: "Learn to learn". Its goal is to know the similarity and difference between objects.

Support set is meta learning's jargon. The small set of labeled images is called a support set. The difference between the training and the support set is the training set is big and the support set is small. The support set can only provide additional information at test time.

Few-shot learning is a kind of meta-learning. Meta learning is different from traditional supervised learning. Tranditional supervised learning asks the model to recognize the training data and then generalize to unseen test data. Differently, meta learning's goal is to learn. In meta-learning, the unknown object is called a query.

In few-shot learning, the query sample is never seen before. The query sample is from an unknown class. This is the main difference from tranditional supervised learning.

K-way means the support set has k classes. N-shot means every class has n samples. The support set is called k-way and n-shot. As the number of ways increases, the prediction accuracy drops. As the number of shots increases, the prediction accuracy improves.

The basic idead of few-shot learning is to train a function that predicts similarity. After training, the learned similarity function can be used for making predictions for unseen queries. We can use the similarity function to compare the query with every sample in the support set and calculate the similarity scores. Then, find the sample with the highest similarity score and use it as the prediction. Given a support set, we can compute the similarity between the query and every sample in the support set to find the most similar sample.

Datasets for few-shot learning; 2 datasets that are most widely used in research papers: Omniglot (hand-written dataset) and Mini-ImageNet.

----------

### Meta Learning | [Wiki](https://en.wikipedia.org/wiki/Meta_learning_(computer_science))
Meta learning is a subfield of machine learning where automatic learning algorithms are applied to metadata about machine learning experiments. The main goal os to use such metadata to understand how automatic learning can become felxible in solving learning problems, hence to improve the performance of existing learning algorithms or to learn (induce) the learning algorithm itself, hence the alternative term learning to learn.

----------

### Zero-Shot Learning | [Article](https://www.analyticsvidhya.com/blog/2022/02/classification-without-training-data-zero-shot-learning-approach/)

In a zero-shot learning approach, we have data in the following manner. Seen classes: classes with labels available for training. Unseen classes: classes that occur only in the test set or during inference and not present during training. Auxiliary information: information about both seen and unseen class labels during training time.

Based on the data available during inference zero-shot learning can be classified into two types. Conventional zero-shot learning: if during test time we only expect images from unseen classes. Generalized zero-shot learning: if during testing phase images from both seen and unseen class can be present.

To evalute a zero-shot recognition model top-1 accuracy metric is generally used. This metric is similar to accuracy but for zero-shot, we take the average accuarcy of both the seen and unseen classes. We want both their accuarcy to be high for this reason Harmonic mean of the accuracy of both classes is selected as a metric. 

Accuracy = 1/N * sum(correct_predictions/total_number_of_samples)

Mean = (2 * Accuracy_on_seen_class * Accuracy_on_unseen_class) / (Accuracy_on_seen_class + Accuracy_on_unseen_class)

This can be thought of as a transfer learning approach where we try to transfer information from target classes.

### One-shot Learning | [Wiki](https://en.wikipedia.org/wiki/One-shot_learning)
One-shot learning is an object categorization problem, found mostly in computer vision. Whereas most machine learning-based object categorization algorithms require training on hundreds of thousands of samples, one-shot learning aims to classify objects from one, or only a few, samples.

----------

## Metric Learning
### *Siamese Neural Networks for One-shot Image Recognition* | [ICML 2015](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) | [Blog (KR)](https://tyami.github.io/deep%20learning/Siamese-neural-networks/)

- Siamese Neural Network | [Wiki](https://en.wikipedia.org/wiki/Siamese_neural_network)
A Siamese neural network (twin NN) uses the same weights while working in tandem on two different input vectors to compute comparable output vectors.

### *Matching Networks for One Shot Learning* | [NIPS 2016](https://proceedings.neurips.cc/paper/2016/file/90e1357833654983612fb05e3ec9148c-Paper.pdf)

### *Prototypical Networks for Few-shot Learning* | [NIPS 2017](https://papers.nips.cc/paper/2017/file/cb8da6767461f2812ae4290eac7cbc42-Paper.pdf)

### *Learning to Compare: Relation Network for Few-Shot Learning* | [CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sung_Learning_to_Compare_CVPR_2018_paper.pdf)

## Graph Neural Network

### *Few-Shot Learning with Graph Neural Networks* | [ICLR 2018](https://openreview.net/pdf?id=BJj6qGbRW) | [arXiv](https://arxiv.org/abs/1711.04043)

### *Learning to Propagate Labels: Transductive Propagation Network for Few-Shot Learning* | [ICLR 2019](https://openreview.net/pdf?id=SyVuRiC5K7) | [arXiv](https://arxiv.org/abs/1805.10002)

### *Edge-Learning Graph Neural Network for Few-shot Learning* | [CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kim_Edge-Labeling_Graph_Neural_Network_for_Few-Shot_Learning_CVPR_2019_paper.pdf) | [arXiv](https://arxiv.org/pdf/1905.01436.pdf)

----------

## Detection
### *LSTD: A Low-Shot Transfer Detector for Object Detection* | [AAAI 2018](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16778/16580)

### *Few-shot Object Detection via Feature Reweighting* | [ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Kang_Few-Shot_Object_Detection_via_Feature_Reweighting_ICCV_2019_paper.pdf)

### *Meta-Learning to Detect Rare Objects* | [ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Meta-Learning_to_Detect_Rare_Objects_ICCV_2019_paper.pdf)

### *One-Shot Object Detection with Co-Attention and Co-Excitation* | [NIPS 2019](https://openreview.net/pdf?id=Hye3UNrlLS) | [GitHub](https://github.com/timy90022/One-Shot-Object-Detection)

----------

### *Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples* | [Google AI Blog](https://ai.googleblog.com/2020/05/announcing-meta-dataset-dataset-of.html) | [ICLR 2020](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/viewer.html?pdfurl=https%3A%2F%2Fopenreview.net%2Fattachment%3Fid%3DrkgAGAVKPr%26name%3Doriginal_pdf&clen=1281351)

---

### Reference
- Meta Learning Wiki, https://en.wikipedia.org/wiki/Meta_learning_(computer_science), 2021-12-14-Tue.
- Few-Shot Learning Blog KR, https://www.kakaobrain.com/blog/106, 2021-10-19-Tue.
- Edge-Learning Graph Neural Network for Few-shot Learning KR, https://www.kakaobrain.com/blog/112, 2021-12-14-Tue.
- Few-Shot Object Detection Blog KR, https://blog.si-analytics.ai/3, 2021-12-06-Mon.
- Few-Shot Object Detection Blog KR, https://blog.si-analytics.ai/7?category=894440, 2021-12-06-Mon.
- Meta Learning Blog KR, https://rhcsky.tistory.com/5, 2021-12-14-Tue.
- Graph Neural Networks for Few-shot and Zero-shot Learning GitHub, https://github.com/thunlp/GNNPapers#few-shot-and-zero-shot-learning, 2021-12-20-Mon.
- Meta Dataset Google AI Blog, https://ai.googleblog.com/2020/05/announcing-meta-dataset-dataset-of.html, 2022-01-26-Wed.
- An Introduction to Few-Shot Learning, https://www.analyticsvidhya.com/blog/2021/05/an-introduction-to-few-shot-learning/, 2022-02-23-Wed.
- Classification without Training Data: Zero-shot Learning Approach, https://www.analyticsvidhya.com/blog/2022/02/classification-without-training-data-zero-shot-learning-approach/, 2022-02-23-Wed.
- What is Few-Show Learning? Methods & Applications in 2022, https://research.aimultiple.com/few-shot-learning/, 2022-02-28-Mon.
- One-shot Learning WiKi, https://en.wikipedia.org/wiki/One-shot_learning, 2022-06-29-Wed.
- Siameses Network Blog KR, https://tyami.github.io/deep%20learning/Siamese-neural-networks/, 2022-09-30-Fri.
