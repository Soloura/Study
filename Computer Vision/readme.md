# Computer Vision
### LeNet | [Paper (Homepage)](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
#### Gradient-Based Learning Applied to Document Recognition
Yann LeCun 1998에 발표한 모델로, CNN을 도입했으며 우편 번호, 숫자를 인식한다.

### AlexNet | [Paper (NIPS)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
#### ImageNet Classification with Deep Convolutional Neural Networks
2012년 ImageNet ILSVRC에서 1위를 하며 CNN을 널리 알리게 된 모델로, 주로 convolutional layer 다음에 pooling layer가 오는 구조와 달리 convolutional layer가 오도록 구성했다.

### ZFNet | [Paper (Homepage)](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf)
#### Visualizing and Understanding Convolutional Networks
2013년 ImageNet ILSVRC에서 1위를 한 모델로, NYU의 Matthew Zeiler와 Rob Fergus의 성 앞글자를 하나씩 따서 이름이 붙었고, 중간 convolutional layer의 크기를 늘린 구조이다.

### NIN | [Paper (arXiv)](https://arxiv.org/pdf/1312.4400.pdf)
#### Network In Network
네트워크 속의 네트워크로, 기존의 CNN의 linear convolution layer와 달리 filter 대신에 MLP(Multi-Layer Perceptron)을 사용하며 non-linear한 성질을 이용해서 feature 추출을 한다. MLP Convolutional layer 여러 개를 network로 만들어서 사용하기 때문에 network in network이다. 또한 1x1 convolution을 이용하여 feature map을 줄였다. 1x1 convolution은 neurons that fire together, wire together인 Hebbian principle와 같이 여러 개의 feature map에서 비슷한 성질을 묶을 수 있어 숫자를 줄여 연산량을 줄일 수 있다. 그리고 기존 CNN와 달리 마지막 layer에 fully connected layer가 아닌 global average pooling을 classifier로 사용하여 overfitting과 연산을 줄인다.

### Auxiliary Classifier | [Paper (arXiv)](https://arxiv.org/pdf/1505.02496.pdf)
#### Training Deeper Convolutional Networks with Deep SuperVision
Auxiliary Classifier block을 이용하면 backpropagation 때 결과를 합치기에 gradient가 작아지는 문제를 피할 수 있다.

### GoogLeNet | [Paper (arXiv)](https://arxiv.org/pdf/1409.4842.pdf)
#### Going deeper with convolutions
2014년 ImageNet ILSVRC에서 1위한 Google에서 만든 모델로, Inception module의 개념을 만들었으며, 이를 통해 parameter를 AlexNet 60M에서 GoogLeNet을 4M으로 줄였다. 1x1 convolution, NIN, Inception module을 사용하여 연산량을 유지하면서 network를 깊고 넓게 만들었다. Auxiliary classifier block unit을 통해 vanishing gradient를 피한다. 

### VGGNet | [Paper](https://arxiv.org/pdf/1409.1556.pdf)
#### Very Deep Convolutional Networks for Large-Scale Image Recognition
2014년 ImageNet ILSVRC에서 2위한 Oxford University에서 만든 모델로 depth에 따른 영향을 나타냈다. 시작부터 끝까지 3x3 convolution과 2x2 max pooling을 사용하는 homogeneous 구조에서 depth가 16일 때 최적의 결과가 나오며, 분류 성능은 GoogLeNet에 비해 성능은 부족하지만 다중 전달 학습 과제에서는 성능이 우월했다. 메모리, parameter가 크다는 단점이 있다.

### ResNet | [Paper (CVPR)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
#### Deep Residual Learning for Image Recognition
2015년 ImageNet ILSVRC에서 1위를 한 Microsoft에서 만든 모델로, layer 수가 Deep 보다 많은 Deeper한 네트워크에 대해서 학습을 하는 residual framework/module을 소개했다.

### DenseNet | [Paper (IEEE)](https://ieeexplore.ieee.org/document/8099726) | [Paper (arXiv)](https://arxiv.org/pdf/1608.06993.pdf)
#### Densely Connected Convolutional Networks
Huang이 제안한 ResNet의 확장판으로 ResNet 블록에서는 합산을 통해 이전 layer와 현재 layer가 합쳐졌다. DenseNet의 경우, 연결을 통해 합쳐진다. 모든 layer를 이전 layer와 연결하고 현재 layer를 다음 layer에 연결한다. 이를 통해 더 매끄러운 기울기, 특징 변환 등과 같은 여러 가지 이점을 제공한다. 또한 parameter의 개수가 줄어든다.

### MobileNet

### MnasNet | [Paper (arXiv)](https://arxiv.org/pdf/1807.11626.pdf)

### EfficientNets | [Paper (arXiv)](https://arxiv.org/pdf/1905.11946.pdf) | [GitHub](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)

### TecoGAN | [Paper (arXiv)](https://arxiv.org/pdf/1811.09393.pdf) | [GitHub](https://github.com/thunil/TecoGAN)
#### Learning Temporal Coherence via Self-Supervision for GAN-based Video Generation

### SinGAN | [Paper (arXiv)](https://arxiv.org/pdf/1905.01164.pdf)
#### SinGan: Learning a Generative Model from a Single Natural Image
SinGAN은 InGan과 마찬가지로 a single natural image로 부터 여러 image를 생성하는 연구이지만, 차이점은 InGAN은 a single image에 대해서 여러 condition을 적용했지만, SinGAN은 unconditional한 방식이다.

### InGAN | [Paper (ICCV)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Shocher_InGAN_Capturing_and_Retargeting_the_DNA_of_a_Natural_Image_ICCV_2019_paper.pdf) | [Paper (arXiv)](https://arxiv.org/abs/1812.00231)
#### InGAN: Capturing and Retargeting the "DNA" of a Natural Image

#### Reference
- Blog (KR-KO): [Laon People Machine Learning Academy](https://blog.naver.com/laonple/220463627091)
- Book (KR-KO): [Deep Learning for Computer Vision](http://www.yes24.com/Product/Goods/63830791)
