# :eyeglasses: Computer Vision

## :cloud: Dehazing

Haze, fog를 제거하는 내용이다. 대상 물체와 관찰자 사이에 존재하는 대기 물질들에 의해 빛의 진행이 방해를 받아 대상이 뿌옇게 보인다.

### :cloud: Single Image Haze Removal Using Dark Channel Prior | [CVPR, 2009](http://mmlab.ie.cuhk.edu.hk/archive/2009/dehaze_cvpr2009.pdf) | [Blog (KR)](https://hyeongminlee.github.io/post/pr001_dehazing/)

Haze가 존재하지 않는 픽셀들은 대부분 RGB의 3 채널 중 적어도 1 채널의 명도 값이 매우 낮은 경향이 있다. 이 채널을 dark channel이라 정의하고 사람이 관찰을 통해 얻어낸 사전 지식 prior를 통해 haze를 제거하는 알고리즘을 제안한다.

보통 haze가 존재하지 않는 야외 이미지는 colorful하거나 어두운 색의 대상들 또는 그들의 그림자로 가득 차 있기 문에 dark pixel 값이 작게 나온다. 반면 haze가 존재하는 이미지는 대상들이 뿌옇고 잘 보이지 않기 때문에 dark pixel 값이 크게 나온다. 이를 통해 이미지에서 haze를 분리할 수 있다. Dark channel prior를 통해 원본 이미지에서 transmission map을 구하고, soft matting을 적용하여 block 현상을 제거하고, 원본 이미지에서 haze를 제거한 뒤, 마지막으로 가장 밝은 airlight를 구한 뒤 depth map 추정까지 한다.

단, 다양한 이미지들을 보고 통계적으로 얻은 prior를 사용하기 때문에 특정 이미지에 대해서는 haze removal이 잘 되지 않을 수 있다. 또한 object가 대기의 빛과 비슷한 색상을 띠면서 그림자마다 없다면 haze로 취급될 수 있다.

## :telescope: Depth Prediction

It is a field that calculates a depth map from a given image or video(or construct a three-dimensional space).

주어진 RGB 이미지에서 depth map을 계산하는 내용이다.

### :telescope: Deeper Depth Prediction with Fully Convolutional Residual Networks | 3DV, 2016 | [arXiv](https://arxiv.org/abs/1606.00373) | [GitHub](https://github.com/irolaina/FCRN-DepthPrediction)

Depth map을 예측하는 fully convolutional architecture를 제안한다. Residual learning, feature map up-sampling, reverse Huber loss function을 이용해서 기존의 방법들에 비해 더 적은 parameters, 실시간 연산, 더 좋은 성능을 가진다.

Unpooling layer, kernel, ReLU로 up-convolution block을 만들고, 반대의 개념으로 up-sampling res-block을 만들었고, 이를 up-projection이라 이름 붙였다. 이를 통해 convolutional layer를 지날수록 resolution이 작아지는 걸 다시 키우고, depth prediction을 가능하게 했다. 그리고 이를 reformulate하여 훈련 시간을 줄이고 효율을 높였다. Potentially non-zero values에 대해서만 계산을 유도하도록 경험/직관적으로 unpooling이 75% 되었을 때 하도록 reformulate하였다.

## :ballot_box_with_check: Object Recognition | [MathWorks (KR)](https://kr.mathworks.com/solutions/image-video-processing/object-recognition.html)

객체 인식은 이미지 또는 비디오 상의 객체를 식별하는 컴퓨터 비전 기술입니다. 객체 인식은 딥러닝과 머신 러닝 알고리즘을 통해 산출되는 핵심 기술입니다. 사람은 사진 또는 비디오를 볼 때 인물, 물체, 장면 및 시각적 세부 사항을 쉽게 알아챌 수 있습니다. 이 기술의 목표는 이미지에 포함된 사항을 이해하는 수준의 능력과 같이 사람이라면 당연히 할 수 있는 일을 컴퓨터도 할 수 있도록 학습시키는 것입니다.

객체 인식은 무인 자동차에서 활용되는 핵심 기술로, 자동차가 정지 신호를 인식하고 보행자와 가로등을 구별할 수 있도록 합니다. 또한 객체 인식 기술은 바이오이미징에서의 질병 식별, 산업 검사 및 로봇 비전과 같은 여러 분야에서 활용할 수 있습니다.

### Object Recognition vs. Object Detection

객체 감지(Object detection)와 객체 인식(Object recognition)은 서로 유사한 객체 식별 기술이지만, 실행 방식은 서로 다르다. 객체 감지는 이미지에서 객체의 인스턴스를 찾아내는 프로세스이다. 객체 감지는 이미지에서 객체를 식별할 뿐 아니라 위치까지 파악되는 객체 인식의 서브셋이다. 이를 통해 하나의 이미지에서 여러 객체를 식별하고 각 위치를 파악할 수 있다.

### Haar-like Feature | [WiKi](https://en.wikipedia.org/wiki/Haar-like_feature)

Haar-like features are digital image features used in object recognition. They owe their name to their intuitive similarity with Haar wavelets and were used in the first real-time face detector.

### :art: Does Colour Really Matter? Evaluation via Object Classification | [CIC 2018](https://www2.cs.sfu.ca/~funt/Funt_Zhu_DoesColourMatter_CIC26_2018.pdf)

ResNet-50 trained with color images performed better in object classification by 12% than that of gray images.

For some tasks, the model trained with color images succeeded, but the model trained with gray images failed.

### LeNet: Gradient-Based Learning Applied to Document Recognition | [Proceedings of the IEEE 1998](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)

LeNet은 Yann LeCun이 제안한 CNN 모델이다. LeNet은 손글씨로 된 우편 번호(숫자)를 인식한다.

LeNet은 기존의 Fully Connected(FC/Dense) layer를 개선하고자 연구되었다. Image는 spatial structure, information을 갖는데, FC layer에 통과시키기 위해 flatten 작업을 거치면 topology data를 잃게 된다. LeNet은 local receptive field, shared weight, subsampling을 결합한 convoltuional layer을 이용한다.

LeNet-1부터 LeNet-5이 연구 및 개발되었는데, 차이는 convolution kernel/filter의 개수를 늘리고 마지막 FC layer 크기를 키웠다. LeNet-1은 input-convolution-subsampling-convolution-subsampling-convolution-output이다. LeNet-5는 Input-C1(Convolution)-S2(Subsampling)-C3(Convolution)-S4(Subsampling)-C5(Full connection)-F6(Full connection)-Output(Gaussian connection)이다.

### AlexNet: ImageNet Classification with Deep Convolutional Neural Networks | [NIPS 2012](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

AlexNet은 2012년 ImageNet ILSVRC에서 1위를 하며 CNN을 널리 알렸다.

AlexNet은 주로 convolutional layer 다음에 pooling layer가 오는 구조와 달리 convolutional layer가 오도록 구성했다.

### ZFNet: Visualizing and Understanding Convolutional Networks | [ECCV 2014](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf)

ZFNet은 2013년 ImageNet ILSVRC에서 1위를 한 모델로, NYU의 Matthew Zeiler와 Rob Fergus의 성 앞글자를 하나씩 따서 이름이 붙었다.

ZFNet은 중간 convolutional layer의 크기를 늘린 구조이다.

### GoogLeNet: Going deeper with convolutions | [CVPR 2015](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf) | [arXiv 2014](https://arxiv.org/abs/1409.4842)

GoogLeNet은 2014년 ImageNet ILSVRC에서 1위한 모델이다.

GoogLeNet은 Inception module을 만들었으며, 이를 통해 parameter를 60M(AlexNet)에서 4M(GoogLeNet)으로 줄였다. GoogLeNet은 1x1 convolution, NIN, Inception module을 사용하여 연산량을 유지하면서 network를 깊고 넓게 만들었다. GoogLeNet은 Auxiliary classifier block unit을 통해 vanishing gradient를 피한다. 

### VGGNet: Very Deep Convolutional Networks for Large-Scale Image Recognition | [ICLR 2015](https://www.robots.ox.ac.uk/~vgg/publications/2015/Simonyan15/) | [arXiv, 2014](https://arxiv.org/abs/1409.1556)

VGGNet은 2014년 ImageNet ILSVRC에서 2위한 Oxford University에서 만든 모델로 depth에 따른 영향을 나타낸다.

VGGNet은 시작부터 끝까지 3x3 convolution과 2x2 max pooling을 사용하는 homogeneous 구조에서 depth가 16일 때 최적의 결과가 나오며, 분류 성능은 GoogLeNet에 비해 성능은 부족하지만 다중 전달 학습 과제에서는 성능이 우월했다. VGGNet은 메모리, parameter가 크다는 단점이 있다.

### ResNet: Deep Residual Learning for Image Recognition | [CVPR 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)

ResNet은 2015년 ImageNet ILSVRC에서 1위를 한 Microsoft에서 만든 모델이다.

ResNet은 layer 수가 Deep 보다 많은 Deeper한 네트워크에 대해서 학습을 하는 residual framework/module을 사용한다.

### ResNeXt: Aggregated Residual Transformations for Deep Neural Networks | [CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.pdf) | [arXiv](https://arxiv.org/abs/1611.05431)

ResNeXt는 2016년 UCSD와 Facebook에서 제안한 ResNet의 변형 network이다.

Input을 _group convolution_을 통해 여러개로 나누고 1x1 convolution으로 input을 transform하고, concat를 통해 merge한다(*Split-Transform-Merge*). ResNet에 비해 *parameter를 줄여 연산량을 줄이고* 더 높은 성능을 보였다. Network에서 각 convolutional layer를 지날 때 마다 output의 크기가 1/2로 줄어든다. ResNet은 하나의 convolutional layer을 통해 deep하게 만들었지만, ResNeXt는 조금 더 깊지만 group convolution을 통해 연산량을 낮췄다. 논문에서 나온 C는 *cardinarity*로 group convolution의 수(the size of the set of transformation)이다. 

ResNet에서는 50 이하 depth일 때는 block 1개, convolution을 2개만 연산했다. 하지만 ResNeXt에서는 2개의 block은 group convolution의 효과가 없어서 block depth가 3 이상일 때부터 효과가 있다. Cardinality의 크기를 키울수록(group 수가 많아질수록) parameter를 줄여 연산량을 줄일 수 있다. 즉, 같은 parameter일 때 더 많은 channel 이용해서 deeper network 설계가 가능하다.

### DenseNet: Densely Connected Convolutional Networks | [CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf) | [arXiv 2016](https://arxiv.org/abs/1608.06993)

DenseNet은 Huang이 제안한 ResNet의 확장 버전으로 ResNet 블록에서는 합산을 통해 이전 layer와 현재 layer가 합쳐졌다.

하지만, DenseNet의 경우, 연결을 통해 합쳐진다. 모든 layer를 이전 layer와 연결하고 현재 layer를 다음 layer에 연결한다. 이를 통해, 더 매끄러운 기울기, 특징 변환 등과 같은 여러 가지 이점을 제공한다. 또한 parameter의 개수가 줄어든다.

### MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications | [arXiv 2017](https://arxiv.org/abs/1704.04861)

MobileNet은 Google에서 연구한 Network로 version 1, 2, 3은 각 2017, 2018, 2019에 발표되었다.

고성능의 device가 아닌 vehicle, drone, smart phone과 같은 환경에서는 computing power, memory가 부족하다. 따라서 battery performance가 중요한 곳을 목적으로 설계된 CNN이다. 

작은 neural network를 만드는 방법은 다음과 같다:

1. Remove fully-connected layers: CNN parameters 90%를 FC layers가 차지한다.
2. Kernel reduction: 3x3을 1x1으로 변경해서 연산량을 줄인다. 
3. Channel reduction. 
4. Evenly spaced downsampling: 초반에 downsampling을 많이 하면 accuracy가 떨어지지만 parameter가 적어지고, 후반에 downsampling을 많이 하면 accuracy가 좋아지지만 parameter가 많아지기 때문에 적당히 사용하는 것이 좋다. 
5. **Depthwise seperable convolutions.**
6. Shuffle operations. 
7. Distillation & Compression.

기존의 CNN은 HxW 크기의 C개의 채널 image에 KxKxC 크기의 M개 filter를 곱하여 H'xW' 크기의 M 채널의 image를 생성한다. Depthwise & Pointwise convolution은 이와 달리 한 방향으로만 크기를 줄이는 전략이다. Depthwise convolution은 channel 개수는 줄어들지 않고 1개의 channel에서의 **크기**만 줄어든다. Pointwise convolution은 **channel** 숫자가 1개로 줄어든다. 기존 CNN의 parameter 수는 K^2CM, 계산량은 K^2CMH'W'이다. Depthwise convoltuion과 Pointwise convolution을 합한 parameter는 K^2C+CM, 계산량은 K^2CW'H' + CMW"H"이다. 만약 W'=W", H'=H"이면 W'H'C(K^2+M)이다. 즉, Depthwise convolution과 pointwise convolution을 합친 Separable convolutions의 계산량은 기존 CNN에 비해서 (1/M + 1/K^2)으로 K=3일 경우 약 8~9배의 효율을 보인다. (H=H'=H", W=W'=W"d 일 때)

### MobileNetV2: Inverted Residuals and Linear Bottlenecks | [CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.pdf) | [arXiv](https://arxiv.org/abs/1801.04381) | [Blog (KR)](https://gaussian37.github.io/dl-concept-mobilenet_v2/)

stride 1 residual block, stride 2 block, Linear Bottlenecks, Inverted Residuals: Bottleneck Residual Block

### Searching for MobileNetV3 | [ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf) | [Blog (KR)](https://seongkyun.github.io/papers/2019/12/03/mbv3/)

Keyword: Small, Large, MnasNet, NetAdapt, Hard-Swish, SE block

### Squeeze-and-Excitation Networks | [CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf) | [arXiv](https://arxiv.org/pdf/1709.01507.pdf) | [Blog (KR)](https://jayhey.github.io/deep%20learning/2018/07/18/SENet/)

A semi-supervised learning method. Like Pseudo Labels, Meta Pseudo Labels as a teacher network to generate pseudo labels on unlabeled data to teach a student network. However, unlike Pseudo Labels where the teacher is fixed, the teacher in Meta Pseudo Labels is constantly adapted by the feedback of the student's performance on the labeled dataset. As a result, the teacher generates better pseudo labels to teach the student.

### Self-training with Noisy Student improves ImageNet classification | [2020 CVPR](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xie_Self-Training_With_Noisy_Student_Improves_ImageNet_Classification_CVPR_2020_paper.pdf)

First train an EfficientNet model on labeled ImageNet images and use it as a teacher to generate pseudo labels on 300M unlabeled images. Then traina larger EfficientNet as a student model on the combination of labeled and pseudo labeled images. Iterate this process by putting back the student as the teacher. During the generation of the pseudo labels, the teacher is not noised so that the pseudo labels are as accurate as possible. However, during the learning of the student, we inject noise such as dropout, stochasitc depth and data augmentation via RandAugment to the student so that the student generalizes better than the teacher.

### Meta Pseudo Labels | [2021 CVPR](https://openaccess.thecvf.com/content/CVPR2021/papers/Pham_Meta_Pseudo_Labels_CVPR_2021_paper.pdf)

[Blog (KR)](https://kmhana.tistory.com/33)

## :microscope: Object Detection

### AdaBoost (Adaptive Boosting) | [Blog #1 (KR)](https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-14-AdaBoost) [Blog #2 (KR)](https://yngie-c.github.io/machine%20learning/2021/03/20/adaboost/)

Adaptive Boosting is a weak learner which uses several stumps called forest of stumps with different weights (amount of say). Each stumps effect after stumps sequentially. AdaBoost learns faster than bagging methods because the number and complexity of calculation of boosting is less than those of bagging methods.

### Haar-Cascade: Rapid Object Detection using a Boosted Cascade of Simple Features | [CVPR, 2001](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf) | [Blog (KR)](https://darkpgmr.tistory.com/116)

“Haar-Cascade” uses elementary features using brightness differences between regions as features. The Haar-Cascade method extracts the features of an object by combining these elementary features by position and size. Therefore, it is useful when detecting objects such as human faces or stairs with clear patterns. The name “Haar” is from the “Haar” features, and “Cascade” is from cascade classifier “AdaBoost”.

영상에서 영역과 영역의 밝기 차이를 특징으로 이용한 다양한 형태의 elementary features가 있으며, 이들을 다양한 위치와 크기로 조합하여 물체에 대한 특징을 추출하는 알고리즘이다. 

Edge, line and center-surround features 등에 대한 특징 값은 feature의 흰색 부분에 해당하는 영상 픽셀들의 밝기 합에서 검은색 부분의 밝기 합을 뺀 차로 계산한다. 그리고 계산된 영역의 밝기 차이가 feature의 threshold 값과 비교를 통해 대상을 식별한다. Multiple features를 사용하며 대상 물체에 대한 조합을 만들어 만족하면 대상이고 만족하지 않으면 배경이라 판단한다.

같은 종류의 feature이여도 물체 내에서의 위치 및 크기에 따라 서로 다른 feature로 간주하기 때문에 다양한 feature 조합이 가능하다. 다양한 features 중 대상과 관련이 있는 의미 있는 feature 선정은 boosting 알고리즘 등의 학습 알고리즘을 이용한다.

물체의 기하학적인 정보를 유지하며 영역 단위의 밝기 차이를 이용하기 때문에 영역 내부에서의 물체의 형태 변화 및 약간의 위치 변화를 어느 정도 커버할 수 있다. 하지만 영상의 contrast, 광원의 방향에 따른 영상 밝기의 변화에 영향을 받으며 물체가 회전된 경우에는 object detection이 힘들다.

Haar-like Feature, Integral Image, AdaBoost(Ensemble, Bagging, Boosting, Stump, weak learner, Amount of Say), Cascade Classifier

### SIFT: Distinctive Image Features from Scale-Invariant Keypoints | [IJCV, 2004](https://people.eecs.berkeley.edu/~malik/cs294/lowe-ijcv04.pdf) | [Blog (KR)](https://darkpgmr.tistory.com/116)

식별하기 쉬운 특징점들을 선택한 뒤, 각 특징점을 중심으로 local patch에 대해 특징 벡터를 추출한 feature이다. 

SIFT feature vector는 특징점 주변의 영상 패치를 4x4 블록으로 나눈뒤, 각 블록에 속한 픽셀들의 gradient 방향과 크기에 대한 히스토그램을 구한 후 이 히스토그램 binary 값들을 일렬로 쭉 연결한 128차원의 벡터다.

SIFT는 기본적으로 특징점 주변의 local gradient 분포 특성(밝기 변화의 방향 및 밝기 변화의 급격한 정도)을 표한하는 feature다.

### HOG: Histograms of Oriented Gradients for Human Detection | [CVPR, 2005](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf) | [Blog (KR)](https://darkpgmr.tistory.com/116)

“Histogram of Gradient” is a vector that divides an image into cells of a certain size, obtains a histogram of the directions of pixels (edge pixels) whose gradient magnitude is greater than a certain value for each cell, and connects these values. HOG is a combination of edge-based template matching and histogram-based matching methods. Therefore, HOG maintains geometric information on a block-by-block basis and uses a histogram inside a block to be robust to changes. As a result, the HOG is suitable for object detection when the shape of the object is not significantly changed, the inner pattern is simple, and it can be identified by the outline such as the case of a person.

영상에서 대상 영역을 일정 크기의 셀로 나누고, 각 셀마다 gradient 크기가 일정 값 이상인 edge 픽셀들의 방향에 대한 히스토그램을 구한 후, 이들 히스토그램 binary 값들을 일렬로 연결한 벡터이다.

HOG는 Edge의 방향 히스토그램 템플릿이다. 템플릿 매칭의 경우 영상의 기하학적 정보를 그대로 유지하여 매칭할 수 있지만 형태나 위치가 조금만 바뀌어도 매칭이 잘 안되는 문제가 있다. 반면에 히스토그램 매칭은 대상의 형태가 변해도 매칭을 할 수 있지만 대상의 기하학적 정보를 잃어버리고 단지 분포 정보만 기억하기 때문에 잘못된 대상과도 매칭이 되는 문제가 있다. HOG는 템플릿 매칭과 히스토그램 매칭의 중간 단계의 방법이며 블록 단위로는 기하학적 정보를 유지하되, 각 블록 내부에서는 히스토그램을 사용함으로써 로컬한 변화에는 어느정도 robust하다. 그리고 HOG는 edge 방향 정보를 이용하기 때문에 일종의 edge-based 템플릿 매칭 방법으로 볼 수 있다. Edge는 기본적으로 영상의 밝기 변화, 조명 변화 등에 덜 민감하다. 또한 HOG는 물체의 실루엣 정보를 이용하므로 사람, 자동차 등과 같이 내부 패턴이 복잡하지 않으면서도 고유의 독특한 윤곽선 정보를 갖는 물체를 식별하는데 적합한 영상 feature이다.

Local feature인 SIFT와 비교해보면 HOG는 일종의 템플릿 매칭이기 때문에 물체가 회전된 경우나 형태 변화가 심한 경우에는 검출이 힘들지만 SIFT는 모델의 특징점과 입력 영상의 특징점에 대해 특징점 단위로 매칭이 이루어지기 때문에 물체의 형태 변화, 크기 변화, 회전 등에 무관하게 매칭이 이루어질 수 있다. 이러한 특성에 비추어 보았을 때, HOG는 물체의 형태 변화가 심하지 않고 내부 패턴이 단순하며 물체의 윤곽선으로 물체를 식별할 수 있을 경우에 적합하고 SIFT는 액자 그림과 같이 내부 패턴이 복잡하여 특징점이 풍부한 경우에 적합한 방법이다.

### R-CNN: Rich feature hierarchies for accurate object detection and semantic segmentation | [arXiv](https://arxiv.org/abs/1311.2524)

R-CNN은 2013년 UC Berkeley의 Ross Girshick이 발표한 object detection, semantic segmentation model이다.

이미지를 분류하는 것보다 이미지 안에 object인지 구분하는 것이 어려운 작업이다. R-CNN은 이를 위해 몇 단계를 거친다:
- 먼저 후보 이미지 영역을 찾아내는 region proposal/bounding box를 찾는 단계가 있다.
  - Bounding box를 찾기 위해 색상이나 패턴 등이 비슷한 인접한 픽셀을 합치는 selective search 과정을 거친다.
- 다음 추출한 bounding box를 CNN의 입력으로 넣기 위해 강제로 사이즈를 통일 시킨다.
  - 이 때 CNN은 훈련된 AlexNet의 변형된 버전이다.
  - CNN의 마지막 단계에서 Support Vector Machine(SVM)을 사용하여 이미지를 분류한다.
- 그리고 최종적으로 분류된 object의 bounding box 좌표를 더 정확히 맞추기 위해 linear regression model을 사용한다.

### Fast R-CNN | [arXiv](https://arxiv.org/abs/1504.08083)

Fast R-CNN은 2015년 Microsoft의 Ross Girshick이 ICCV15에서 발표한 R-CNN을 개선한 model이다.

R-CNN의 문제점은 모든 bounding box에 대해 CNN, SVM, linear regression 3가지 모델을 훈련시켜야하기 떄문에 어렵다. 

때문에 Fast R-CNN은 bounding box 사이에 겹치는 영역이 CNN을 통과시키는 것은 낭비라 생각했다.

Region of Interset Pooling(RolPool)의 개념을 도입하여 selective search에서 찾은 bounding box 정보를 CNN을 통과시키면서 유지시키고 최종 CNN feature map으로부터 해당 영역을 추출하여 pooling한다. 이를 통해 bounding box마다 CNN을 거치는 시간을 단축시킨다. 또한 SVM 대신 CNN 뒤에 softmax를 놓고 linear regression 대신 softmax layer와 동일하게 뒤에 추가했다. 

Joint the feature extractor, classifier, regressor together in a unified framework.

### Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks | [arXiv](https://arxiv.org/abs/1506.01497) | [PyCaffe](https://github.com/rbgirshick/py-faster-rcnn) | [PyTorch](https://github.com/longcw/faster_rcnn_pytorch) | [MatLab](https://github.com/ShaoqingRen/faster_rcnn)

Faster R-CNN은 2015년 Microsoft에서 인턴을 했던 USTC의 Shaoqing Ren이 NIPS15에서 발표한 Fast R-CNN의 개선된 model이다. 

Fast R-CNN에서 남은 bottleneck은 bounding box를 만드는 region proposal 단계이다. 

Faster R-CNN은 region proposal 단계를 CNN 안에 넣어서 문제를 해결했다.

CNN을 통과한 feature map에서 sliding window를 이용해 각 anchor마다 가능한 bounding box의 좌표와 이 bounding box의 점수를 계산한다.

대부분 너무 홀쭉하거나 넓은 물체는 많지 않으므로 2:1, 1:1, 1:2 등의 몇가지 타입으로도 좋다.

### YOLO: You Only Look Once, Real-Time Object Detection | [Homepage](https://pjreddie.com/darknet/yolo/)

YOLO는 이미지 내의 bounding box와 class probaility를 single regression problem으로 unified하여 이미지를 한번 보는 것으로 object의 종류와 위치를 추측한다. Single convolutional network를 통해 multiple bounding box에 대한 class probability를 계산하는 방식이다. 기존의 object detection method와 비교했을 때, YOLO의 상대적인 장점과 단점은 다음과 같다. 장점으로는 1. 간단한 처리과정으로 속도가 매우 빠르고 기존의 다른 real-time detection system들과 비교할 때 2배 정도 높은 mAP를 보인다. 2. Image 전체를 1번에 바라보는 방식으로 class에 대한 맥락적 이해도가 높아 낮은 background error(false-negative)를 보인다. 3. Object에 대한 좀 더 일반화된 특징을 학습하기 때문에 natural image로 학습하고 artwork에 테스트해도 다른 detection system에 비해 훨씬 높은 성능을 보인다. 단점으로는 상대적으로 정확도가 낮은데, 특히 작은 object일 수록이다.

Unified Detection은 input image를 S by S grid로 나눈다. 각각의 grid cell은 B개의 bounding box와 각 bounding box에 대한 confidence score를 갖는다. 만약 cell에 object가 없다면 confidence score는 0이 된다. 각각의 grid cell은 C개의 conditional class probability를 갖는다. 각각의 bounding box는 x, y, w, h, confidence로 구성된다. (x, y)는 bounding box의 중심점을 의미하며 grid cell의 범위에 대한 상대값이 입력된다. (w, h) 전체 image의 width, height에 대한 상대값이 입력된다. Test time에는 conditional class probability와 bounding box의 confidence score를 곱하여 class-specific confidence score를 얻는다. 논문에서는 YOLO의 성능평가를 위해 PASCAL VOC를 사용했으며 S, B, C에는 각각 7, 2, 20을 사용했다.

Network Design/Architecture은 GoogLeNet 모델의 24 convolutional layers and 2 fully connected layers을 기반으로 24 convolutional layers를 9개로 대체했다. 계산을 마치면 총 98개의 class specific confidence score를 얻게 되고, 이에 대해 각 20개의 class를 기준으로 non-maximum suppression을 하여 object에 대한 class 및 bounding box location을 결정한다. 

Loss function은 gird cell의 여러 bounding box 중 ground truth box와의 IOU가 가장 높은 bounding box를 predictor로 설정한다. object가 존재하는 grid cell i의 predictor bounding box j, object가 존재하지 않는 grid cell i의 bounding box j, object가 존재하는 grid cell i을 기호로 사용하고, ground truth box의 중심점이 어떤 grid cell 내부에 위치하면, 그 grid cell에는 object가 존재한다고 여긴다. [참고](https://curt-park.github.io/2017-03-26/yolo/)

YOLO의 한계는 1. 각 grid cell이 하나의 class만을 예측할 수 있으므로 작은 object 여러개가 있는 경우에는 제대로 예측하지 못한다. 2. bounding box의 형태가 training data를 통해서만 학습되므로 새로운 형태의 bounding box의 경우 정확히 예측하지 못한다. 3. 몇 단계의 layer를 거쳐서 나온 feature map을 대상으로 bounding box를 예측하므로 localization이 다소 부정확해지는 경우가 있다.

다른 real time object detection에 비해 높은 mAP를 보여주며 fast YOLO의 경우 가장 빠른 속도이다. Fast R-CNN과 비교하면 훨씬 적은 false positive이다. (low background error) Fast R-CNN과 같이 동작하면 보완하는 역할을 할 수 있다.

### SSD: Single Shot MultiBox Detector | [ECCV 2016](https://link.springer.com/chapter/10.1007/978-3-319-46448-0_2) | [arXiv](https://arxiv.org/abs/1512.02325) | [Blog (KR)](https://herbwood.tistory.com/15)

SSD는 2015년에 UNC의 Wei Liu가 ECCV16에서 발표한 object detection method로, single deep neural network를 이용한다.

Multi-scale feature maps for detection: 끝이 잘린 base network에 convolutional feature layers를 추가했다. 이 layers는 크기를 점차 줄여서 다양한 크기에서 prediction을 한다.

Predicting detection을 하는 convolutional model은 feature layer들(Overfeat and YOLO)과 다르다. **Convolutional predictors for detection**

### NIN: Network In Network | [arXiv](https://arxiv.org/abs/1312.4400)

네트워크 속의 네트워크로, 기존의 CNN의 linear convolution layer와 달리 filter 대신에 MLP(Multi-Layer Perceptron)을 사용하며 non-linear한 성질을 이용해서 feature 추출을 한다. MLP Convolutional layer 여러 개를 network로 만들어서 사용하기 때문에 network in network이다. 또한 1x1 convolution을 이용하여 feature map을 줄였다. 1x1 convolution은 neurons that fire together, wire together인 Hebbian principle와 같이 여러 개의 feature map에서 비슷한 성질을 묶을 수 있어 숫자를 줄여 연산량을 줄일 수 있다. 그리고 기존 CNN와 달리 마지막 layer에 fully connected layer가 아닌 global average pooling을 classifier로 사용하여 overfitting과 연산을 줄인다.

### Auxiliary Classifier: Training Deeper Convolutional Networks with Deep SuperVision | [arXiv](https://arxiv.org/abs/1505.02496)

Auxiliary Classifier block을 이용하면 backpropagation 때 결과를 합치기에 gradient가 작아지는 문제를 피할 수 있다.

### YOLOv2: YOLO9000: Better, Faster, Stronger | [Homepage](https://pjreddie.com/darknet/yolov2/) | [arXiv](https://arxiv.org/abs/1612.08242)

### YOLOv3:AnIncrementalImprovement | [Homepage](https://pjreddie.com/darknet/yolo/) |[PDF](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/viewer.html?pdfurl=https%3A%2F%2Fpjreddie.com%2Fmedia%2Ffiles%2Fpapers%2FYOLOv3.pdf&clen=2241388&chunk=true)

### YOLOv4: Optimal Speed and Accuracy of Object Detection | [arXiv](https://arxiv.org/abs/2004.10934) | [Blog (KR)](https://jetsonaicar.tistory.com/68) | [Blog (KR)](https://jjeamin.github.io/darknet_book/part1_paper/yolov4.html)

### YOLOv5 | [GitHub](https://github.com/ultralytics/yolov5) | [Docs](https://docs.ultralytics.com/)

### YOLOv6 - YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications | [GitHub](https://github.com/meituan/YOLOv6) | [arXiv](https://arxiv.org/abs/2209.02976)

* YOLOv6 v3 - YOLOv6 v3.0: A Full-Scale Reloading | [arXiv](https://arxiv.org/abs/2301.05586)

### YOLOv8 | [GitHub](https://github.com/ultralytics/ultralytics) | [Docs](https://docs.ultralytics.com/)

YOLOv8 supports a full range of vision AI tasks, including detection, segmentation, pose estimation, tracking, and classification.

---

## :scissors: Segmentation

### Bayes Matting

사용자가 정의한 trimap을 바탕으로 투명도를 갖도록 컬러 분포를 모델링한다. 사용자의 안쪽 영역과 바깥 영역 입력의 사용자 입력이 필요하다.

### Graph Cut

Bayes Matting과 trimap, 확률 컬러 모델을 모두 갖는 방법이다.

### "GrabCut" - Interactive Foreground Extraction using Iterated Graph Cuts | [ACMTG, 2012](https://cvg.ethz.ch/teaching/cvl/2012/grabcut-siggraph04.pdf)

Graph Cut을 반복적으로 적용하여 투명도가 적용되지 않은 hard segmentation을 먼저 수행한 뒤, border matting 방법을 적용하여 foreground의 경계 부분에 투명도를 할당한 다음, 나머지 배경 부분은 완전히 투명하게 만드는 방식으로 segmentation을 진행한다.

### U-Net | [Blog (KR)](https://velog.io/@lighthouse97/UNet%EC%9D%98-%EC%9D%B4%ED%95%B4)

### [Segment Anything](https://segment-anything.com/) | [GitHub](https://github.com/facebookresearch/segment-anything) | [ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Kirillov_Segment_Anything_ICCV_2023_paper.pdf)

SAM is a promptable segmentation system with zero-shot generalization to unfamiliar objects and images, without the need for additional training.

* [[2023]A Comprehensive Survey on Segment Anything Model for Vision and Beyond (arXiv)](https://github.com/liliu-avril/Awesome-Segment-Anything)
* [Awesome Segment Anything](https://github.com/Hedlen/awesome-segment-anything): summarize the research progress of Segment Anything in various fields, including papers and projects, etc.
* [SAM Blog KR](https://thecho7.tistory.com/entry/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-Segment-Anything-%EC%84%A4%EB%AA%85-%EC%BD%94%EB%93%9C-%ED%8F%AC%ED%95%A8)

* SA
  * (1/3) Task: zero-shot, prompt/interactive segmentation
  * (2/3) Model(SAM)
    * (A/C) powerful image encoder: masked autoencoder (MAE) pre-trained ViT - `image > image encoder > image embedding (256x64x64)`
    * (B/C) prompt encoder: `point/box/text > prompt encoder > output tokens + prompt tokens (Nx256)`
      * (a/b) sparse prompt: points - positional encodings summed with learned embeddings, boxes - positional encodings summed with learned embeddings, text - clip output
      * (b/b) dense prompt: mask - image embedding convolution
    * (C/C) mask decoder: transformer decoder block - prompt self-attention, cross-attention - prompt-to-image, image-to-prompt
    * Resolving mask ambiguity: 3 masks with focal loss(harder the more weights), dice loss(recall/GT) - linear combination
  * (3/3) Data (SA-1B): 1B masks from 11M imgs
    * (A/C) assisted manual: points by human
    * (B/C) semi-automatic: specific object group
    * (C/C) fully automatic: regular grid - points everything

### [Segment Anything Model 2](https://ai.meta.com/sam2/) | [GitHub](https://github.com/facebookresearch/segment-anything-2) | [HuggingFace](https://huggingface.co/papers/2408.00714)

Segment Anything Model 2 (SAM 2) is a foundation model towards solving promptable visual segmentation in images and videos. Extended SAM to video by considering images as a video with a single frame. The model design is a simple transformer architecture with streaming memory for real-time video processing. A model-in-the-loop data engine, which improves model and data via user interaction, to collect SA-V dataset, the largest video segmentation dataset to date. SAM 2 trained on data provides strong performance across a wide range of tasks and visual domains.

----------

## :fork_and_knife: Semantic Segmentation

### Mask R-CNN | [arXiv](https://arxiv.org/abs/1703.06870) | [PyTorch](https://github.com/felixgwu/mask_rcnn_pytorch) | [TesforFlow](https://github.com/CharlesShang/FastMaskRCNN)

Mask R-CNN은 2017년 Facebook의 Kaimimg He가 ICCV17에서 발표한 분할된 image를 masking하는 model이다. Faster R-CNN에 각 픽셀이 object class에 해당하는지 binary masking하는 분기 network를 추가했다. 정확한 픽셀 위치를 추출하기 위해 CNN을 통과하면서 Rol Pooling에서 rounding하며 발생하는 소숫점 오차를 RoIAlign(2D bilinear interpolation)로 대체해서 감소시켰다.

### EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks | 2019 ICML | [arXiv](https://arxiv.org/abs/1905.11946) | [GitHub](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)

[Blog (KR)](https://kmhana.tistory.com/26?category=839700)

### TecoGAN: Learning Temporal Coherence via Self-Supervision for GAN-based Video Generation | [arXiv](https://arxiv.org/abs/1811.09393) | [GitHub](https://github.com/thunil/TecoGAN)

### SinGAN: Learning a Generative Model from a Single Natural Image | [arXiv](https://arxiv.org/pdf/1905.01164.pdf) | [GitHub](https://github.com/FriedRonaldo/SinGAN)

SinGAN은 InGan과 마찬가지로 a single natural image로 부터 여러 image를 생성하는 연구이지만, 차이점은 InGAN은 a single image에 대해서 여러 condition을 적용했지만, SinGAN은 unconditional한 방식이다.

### InGAN: Capturing and Retargeting the "DNA" of a Natural Image | [ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Shocher_InGAN_Capturing_and_Retargeting_the_DNA_of_a_Natural_Image_ICCV_2019_paper.pdf) | [Paper (arXiv)](https://arxiv.org/abs/1812.00231)

---

## :calendar: Optical Character Recognition

## :calendar: Scene Text Detection (STD)

Regression-based text detectors

Segmentation-based text detectors

End-to-end text detectors

Character-level text detectors

### Character Region Awareness for Text Detection | [CVPR, 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Baek_Character_Region_Awareness_for_Text_Detection_CVPR_2019_paper.pdf)

Ground Truth Label Generation, region score(center of the character), affinity score(center of the space between adjacent characters)

Weakly-Supervised Learning

## :calendar: Scene Text Recognition (STR)

### What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis | [ICCV, 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Baek_What_Is_Wrong_With_Scene_Text_Recognition_Model_Comparisons_Dataset_ICCV_2019_paper.pdf)

This paper proposes a model that produces good performance by comparing, analyzing, and combining datasets and models.

Datasets
- Synthetic datasets
  - MJSynth (MJ): designed for STR, 8.9 M word box images
  - SynthText (ST): originally designed for STD, cropping word boxes used for STR, 5.5 M
- Real-world datasets
  - Regular datasets - horizontally laid out characters that have even spacings between them
    - IIIT5K-Words (IIIT)
    - Street View Text (SVT)
    - ICDAR2003 (IC03)
    - ICDAR2013 (IC13)
  - Irregular datasets - contain harder corner cases for STR, such as curved and arbitrarily rotated or distored texts
    - ICDAR2015 (IC15)
    - SVT Perspective (SP)
    - CUTE80 (CT)

Framework
- Transformation - transforms the input image into the normalized image.
  - Thin-plate spline (TPS) transformation - a variant of the spatial transformation network (STN)
- Feature extraction - CNN abstract abstract an input image and outputs a visual feature map
  - VGG
  - RCNN
  - ResNet
- Sequence modeling - extracted features reshaped to be a sequence of features
  - Bidirectional LSTM (BiLSTM)
  - :key: Rosetta removed the BiLSTM to reduct computational complexity and memory consumption.
- Prediction
  - Connectionist temporal classification (CTC): allows for the prediction of a non-fixed number of a sequence even though a fixed number of the features are given.
  - Attention-based sequence prediction (Attn): automatically captures the information flow within the input sequence to predict the output sequence.

---

- Precision
Precision measures how accurate is your predictions. The percentage of your predictions are correct
*Precision = TP / (TP + FP)*

- Recall
Recall measures how good you find all the positives. 
*Recall = TP / (TP + FN)*

- IoU(Intersection over Union)
IoU measures the overlap between 2 boundaries.
*IoU = area of overlap / area of union*

- AP(Area under curve AUC)

- FLOPS(FLoating point Operations Per Second)

- FLOPs(FLoating point OPerationS)

- MAC(Multiply ACcumulate)

---

## :hammer_and_wrench: Activation Function

`This content has moved to the 'Activation-Function' page.`

## :bow_and_arrow: Few-Shot Learning

`This content has moved to the 'Few-Shot Learning' page.`

## :art: Multimodal Learning

`This content has moved to the 'Multimodal Learning' page.`

## :robot: AutoML

`This content has moved to the 'AutoML' page.`

---

### Reference
- Blog KR: [Laon People Machine Learning Academy](https://blog.naver.com/laonple/220463627091)
- Book KR: [컴퓨터 비전과 딥러닝 (Deep Learning for Computer Vision](http://www.yes24.com/Product/Goods/63830791)
- Book KR: [실전! 텐서플로 2를 활용한 딥러닝 컴퓨터 비전](http://www.yes24.com/Product/Goods/90365150)
- LeNet Blog KR, https://my-coding-footprints.tistory.com/97, 2021-03-10-Wed.
- R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN Blog KR, https://tensorflow.blog/2017/06/05/from-r-cnn-to-mask-r-cnn/, 2021-03-05-Fri.
- R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN Youtube, https://youtu.be/kcPAGIgBGRs, 2021-03-09-Tue.
- YOLO OpenCV, https://docs.opencv.org/master/da/d9d/tutorial_dnn_yolo.html, 2021-03-05-Fri.
- YOLO Blog KR, https://curt-park.github.io/2017-03-26/yolo/, 2021-03-05-Fri.
- YOLO Blog KR, https://medium.com/curg/you-only-look-once-%EB%8B%A4-%EB%8B%A8%EC%A7%80-%ED%95%9C-%EB%B2%88%EB%A7%8C-%EB%B3%B4%EC%95%98%EC%9D%84-%EB%BF%90%EC%9D%B4%EB%9D%BC%EA%B5%AC-bddc8e6238e2, 2021-02-25-Thu.
- MobileNet Version 1 Blog KR, http://melonicedlatte.com/machinelearning/2019/11/01/212800.html, 2021-03-08-Mon.
- MobileNet Version 2 Blog KR, https://blog.naver.com/PostView.nhn?blogId=chacagea&logNo=221692490366&categoryNo=0&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=postView, 2021-03-09-Tue.
- MobileNet Version 1 Blog KR, https://blog.naver.com/chacagea/221582912200, 2021-03-09-Tue.
- MobileNet Version 3 Blog KR, https://seongkyun.github.io/papers/2019/12/03/mbv3/, 2021-12-01-Wed.
- SENet Blog KR, https://jayhey.github.io/deep%20learning/2018/07/18/SENet/, 2021-12-05-Sun.
- Selective Search for Object Recognition Slide, http://www.cs.cornell.edu/courses/cs7670/2014sp/slides/VisionSeminar14.pdf, 2021-03-09-Tue.
- Facebook Research Detectron GitHub, https://github.com/facebookresearch/Detectron, 2021-03-09-Tue.
- Facebook Research Detectron2 GitHub, https://github.com/facebookresearch/detectron2, 2021-03-09-Tue.
- OpenMMLab Detection Toolbox and Benchmark GitHub, https://github.com/open-mmlab/mmdetection, 2021-03-09-Tue.
- ResNeXt Blog KR, https://blog.airlab.re.kr/2019/08/resnext, 2021-03-10-Wed.
- ResNeXt Blog KR, https://everyday-deeplearning.tistory.com/entry/%EC%B4%88-%EA%B0%84%EB%8B%A8-%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0ResNext, 2021-03-10-Wed.
- SSD Blog KR, https://taeu.github.io/paper/deeplearning-paper-ssd/, 2021-03-15-Mon.
- SSD Blog KR, https://cocopambag.tistory.com/15, 2021-03-15-Mon.
- mAP(mean Average Precision) for Object Detection Blog US, https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173, 2021-03-23-Tue.
- GrabCut Blog KR, http://www.navisphere.net/2095/grabcut-interactive-foreground-extraction-using-iterated-graph-cuts/, 2021-09-28-Tue.
- Graph Cut Wiki, https://en.wikipedia.org/wiki/Graph_cuts_in_computer_vision, 2021-09-28-Tue.
- Single Image Haze Removal Using Dark Channel Prior Blog KR, https://hyeongminlee.github.io/post/pr001_dehazing/, 2021-10-05-Tue.
- SIFT, HOG, Haar Cascade Algorithm Blog KR, https://darkpgmr.tistory.com/116, 2021-10-09-Sat.
- Ensemble Bagging Boosting Blog KR, https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-11-%EC%95%99%EC%83%81%EB%B8%94-%ED%95%99%EC%8A%B5-Ensemble-Learning-%EB%B0%B0%EA%B9%85Bagging%EA%B3%BC-%EB%B6%80%EC%8A%A4%ED%8C%85Boosting, 2021-11-19-Fri.
- AdaBoost Blog KR, https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-14-AdaBoost, 2021-11-19-Fri.
- Haar-like Feature Wiki, https://en.wikipedia.org/wiki/Haar-like_feature, 2021-11-19-Fri.
- Haar-like Feature Blog KR, https://wiserloner.tistory.com/1101, 2021-11-19-Fri.
- Haar Cascade Blog KR, https://webnautes.tistory.com/1352, 2021-11-19-Fri.
- FLOPS FLOPs Blog KR, https://ladofa.blogspot.com/2020/08/flops-flops.html, 2021-12-05-Sun.
- FLOPS FLOPs MAC Blog KR, https://bongjasee.tistory.com/3, 2021-12-05-Sun.
- YOLOv4 Blog KR, https://jetsonaicar.tistory.com/68, 2022-01-17-Mon.
- YOLOv4 Related Works Blog KR, https://jjeamin.github.io/darknet_book/part1_paper/yolov4.html, 2022-01-17-Mon.
- YOLOv5 GitHub, https://github.com/ultralytics/yolov5, 2022-01-17-Mon.
- YOLOv5 Docs, https://docs.ultralytics.com/, 2022-01-17-Mon.
- SequeezeSeqV2, Darknet51, Darknet52 Blog KR, https://www.wenyanet.com/opensource/ko/6114b45841db8a44780b2403.html, 2022-01-17-Mon.
- 객체 인식 Mathworks KR, https://kr.mathworks.com/solutions/image-video-processing/object-recognition.html, 2022-10-03-Mon.
- Deeper Depth Prediction with Fully Convolutional Residual Networks GitHub, https://github.com/irolaina/FCRN-DepthPrediction, 2023-01-19-Thu.
- AdaBoost Blog KR, https://yngie-c.github.io/machine%20learning/2021/03/20/adaboost/, 2023-01-25-Wed.
- MobileNetV2(모바일넷 v2), Inverted Residuals and Linear Bottlenecks, https://gaussian37.github.io/dl-concept-mobilenet_v2/, 2023-01-29-Sun.
- SSD Blog KR, https://herbwood.tistory.com/15, 2023-11-13-Mon.
- U-Net Blog KR, https://velog.io/@lighthouse97/UNet%EC%9D%98-%EC%9D%B4%ED%95%B4, 2023-11-13-Mon.
- Self-training with Noisy Student improves ImageNet classification, https://openaccess.thecvf.com/content_CVPR_2020/papers/Xie_Self-Training_With_Noisy_Student_Improves_ImageNet_Classification_CVPR_2020_paper.pdf, 2024-04-21-Sun.
- Meta Pseudo Labels, https://openaccess.thecvf.com/content/CVPR2021/papers/Pham_Meta_Pseudo_Labels_CVPR_2021_paper.pdf, 2024-04-21-Sun.
- Meta Pseudo Labels Blog KR, https://kmhana.tistory.com/33, 2024-04-21-Sun.
- EfficientNet Blog KR, https://kmhana.tistory.com/26?category=839700, 2024-04-21-Sun.
- A Comprehensive Survey on Segment Anything Model for Vision and Beyond, https://github.com/liliu-avril/Awesome-Segment-Anything, 2024-04-29-Mon.
- Segment Anything Model 2, https://ai.meta.com/sam2/, 2024-08-03-Sat.
- YOLOv8 GitHub, https://github.com/ultralytics/ultralytics, 2024-08-04-Sun.
- YOLOv8 Docs, https://docs.ultralytics.com/, 2024-08-04-Sun.
- SAM2 GitHub, https://github.com/facebookresearch/segment-anything-2, 2024-08-05-Mon.
- SAM Blog KR, https://thecho7.tistory.com/entry/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-Segment-Anything-%EC%84%A4%EB%AA%85-%EC%BD%94%EB%93%9C-%ED%8F%AC%ED%95%A8, 2024-08-06-Tue.
- YOLOv6 GitHub, https://github.com/meituan/YOLOv6, 2024-08-10-Sat.
