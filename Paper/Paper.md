# Paper
## Computer Vision
### LeNet (Gradient-Based Learning Applied to Document Recognition) | [Paper](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
Yann LeCun 1998에 발표한 모델로, CNN을 도입했으며 우편 번호, 숫자를 인식한다.

### AlexNet (ImageNet Classification with Deep Convolutional Neural Networks) | [Paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
2012년 ImageNet ILSVRC에서 1위를 하며 CNN을 널리 알리게 된 모델로, 주로 convolutional layer 다음에 pooling layer가 오는 구조와 달리 convolutional layer가 오도록 구성했다.

### ZFNet (Visualizing and Understanding Convolutional Networks) | [Paper](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf)
2013년 ImageNet ILSVRC에서 1위를 한 모델로, NYU의 Matthew Zeiler와 Rob Fergus의 성 앞글자를 하나씩 따서 이름이 붙었고, 중간 convolutional layer의 크기를 늘린 구조이다.

### GoogLeNet (Going deeper with convolutions) | [Paper](https://arxiv.org/pdf/1409.4842.pdf)
2014년 ImageNet ILSVRC에서 1위한 Google에서 만든 모델로, Inception 모듈의 개념을 만들었으며, 이를 통해 parameter를 AlexNet 60M에서 GoogLeNet을 4M으로 줄였다.

### VGGNet (Very Deep Convolutional Networks for Large-Scale Image Recognition) | [Paper](https://arxiv.org/pdf/1409.1556.pdf)
2014년 ImageNet ILSVRC에서 2위한 Oxford University에서 만든 모델로 depth에 따른 영향을 나타냈다. 시작부터 끝까지 3x3 convolution과 2x2 max pooling을 사용하는 homogeneous 구조에서 depth가 16일 때 최적의 결과가 나오며, 분류 성능은 GoogLeNet에 비해 성능은 부족하지만 다중 전달 학습 과제에서는 성능이 우월했다. 메모리, parameter가 크다는 단점이 있다.

### ResNet (Deep Residual Learning for Image Recognition) | [Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
2015년 ImageNet ILSVRC에서 1위를 한 Microsoft에서 만든 모델로, layer 수가 Deep 보다 많은 Deeper한 네트워크에 대해서 학습을 하는 residual framework/module을 소개했다.

#### Reference
- [Laon People Machine Learning Academy](https://blog.naver.com/laonple/220463627091)
