# Computer Vision

## 🌳 Object Detection 🌳

### *LeNet: Gradient-Based Learning Applied to Document Recognition* | [Homepage](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
LeNet은 Yann LeCun이 1998에 제안한 Convolutional Neural Network(CNN) 모델이다. LeNet은 손글씨로 된 우편 번호(숫자)를 인식한다. 기존의 Fully Connected(FC)를 개선하고자 연구되었다. Image는 spatial structure, information을 갖는데, FC layer에 통과시키기 위해 flatten 작업을 거치면 topology data를 잃게 된다. LeNet은 local receptive field, shared weight, sub sampling을 결합한 convoltuional layer을 이용한다. LeNet-1부터 LeNet-5이 연구 및 개발되었는데, 차이는 convolution kernel/filter의 개수를 늘리고 마지막 FC layer 크기를 키웠다. LeNet-1은 input-convolution-subsampling-convolution-subsampling-convolution-output이다. LeNet-5는 Input-C1(Convolution)-S2(Subsampling)-C3(Convolution)-S4(Subsampling)-C5(Full connection)-F6(Full connection)-OUTPUT(Gaussian connection)이다.

### *AlexNet: ImageNet Classification with Deep Convolutional Neural Networks* | [NIPS](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
2012년 ImageNet ILSVRC에서 1위를 하며 CNN을 널리 알리게 된 모델로, 주로 convolutional layer 다음에 pooling layer가 오는 구조와 달리 convolutional layer가 오도록 구성했다.

### *ZFNet: Visualizing and Understanding Convolutional Networks* | [Homepage](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf)
2013년 ImageNet ILSVRC에서 1위를 한 모델로, NYU의 Matthew Zeiler와 Rob Fergus의 성 앞글자를 하나씩 따서 이름이 붙었고, 중간 convolutional layer의 크기를 늘린 구조이다.

### *NIN: Network In Network* | [arXiv](https://arxiv.org/abs/1312.4400)
네트워크 속의 네트워크로, 기존의 CNN의 linear convolution layer와 달리 filter 대신에 MLP(Multi-Layer Perceptron)을 사용하며 non-linear한 성질을 이용해서 feature 추출을 한다. MLP Convolutional layer 여러 개를 network로 만들어서 사용하기 때문에 network in network이다. 또한 1x1 convolution을 이용하여 feature map을 줄였다. 1x1 convolution은 neurons that fire together, wire together인 Hebbian principle와 같이 여러 개의 feature map에서 비슷한 성질을 묶을 수 있어 숫자를 줄여 연산량을 줄일 수 있다. 그리고 기존 CNN와 달리 마지막 layer에 fully connected layer가 아닌 global average pooling을 classifier로 사용하여 overfitting과 연산을 줄인다.

### *Auxiliary Classifier: Training Deeper Convolutional Networks with Deep SuperVision* | [arXiv](https://arxiv.org/abs/1505.02496)
Auxiliary Classifier block을 이용하면 backpropagation 때 결과를 합치기에 gradient가 작아지는 문제를 피할 수 있다.

### *GoogLeNet: Going deeper with convolutions* | [arXiv](https://arxiv.org/abs/1409.4842)
2014년 ImageNet ILSVRC에서 1위한 Google에서 만든 모델로, Inception module의 개념을 만들었으며, 이를 통해 parameter를 AlexNet 60M에서 GoogLeNet을 4M으로 줄였다. 1x1 convolution, NIN, Inception module을 사용하여 연산량을 유지하면서 network를 깊고 넓게 만들었다. Auxiliary classifier block unit을 통해 vanishing gradient를 피한다. 

### *VGGNet: Very Deep Convolutional Networks for Large-Scale Image Recognition* | [arXiv](https://arxiv.org/abs/1409.1556)
2014년 ImageNet ILSVRC에서 2위한 Oxford University에서 만든 모델로 depth에 따른 영향을 나타냈다. 시작부터 끝까지 3x3 convolution과 2x2 max pooling을 사용하는 homogeneous 구조에서 depth가 16일 때 최적의 결과가 나오며, 분류 성능은 GoogLeNet에 비해 성능은 부족하지만 다중 전달 학습 과제에서는 성능이 우월했다. 메모리, parameter가 크다는 단점이 있다.

### *ResNet: Deep Residual Learning for Image Recognition* | [CVPR](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
2015년 ImageNet ILSVRC에서 1위를 한 Microsoft에서 만든 모델로, layer 수가 Deep 보다 많은 Deeper한 네트워크에 대해서 학습을 하는 residual framework/module을 소개했다.

### *ResNeXt, Aggregated Residual Transformations for Deep Neural Networks* | [arXiv](https://arxiv.org/abs/1611.05431)
2016년 UCSD와 Facebook에서 제안한 ResNeXt는 ResNet의 변형 network이다. Input을 *group convolution*을 통해 여러개로 나누고 1x1 convolution으로 input을 transform하고, concat를 통해 merge한다(*Split-Transform-Merge*). ResNet에 비해 *parameter를 줄여 연산량을 줄이고* 더 높은 성능을 보였다. Network에서 각 convolutional layer를 지날 때 마다 output의 크기가 1/2로 줄어든다. ResNet은 하나의 convolutional layer을 통해 deep하게 만들었지만, ResNeXt는 조금 더 깊지만 group convolution을 통해 연산량을 낮췄다. 논문에서 나온 C는 *cardinarity*로 group convolution의 수(the size of the set of transformation)이다. ResNet엣는 50 이하 depth일 때는 block 1개, convolution을 2개만 연산했다. 하지만 ResNeXt에서는 2개의 block은 group convolution의 효과가 없어서 block depth가 3 이상일 때부터 효과가 있다. Cardinality의 크기를 키울수록(group 수가 많아질수록) parameter를 줄여 연산량을 줄일 수 있다. 즉, 같은 parameter일 때 더 많은 channel 이용해서 deeper network 설계가 가능하다.

### *DenseNet: Densely Connected Convolutional Networks* | [IEEE](https://ieeexplore.ieee.org/document/8099726) | [arXiv](https://arxiv.org/abs/1608.06993)
Huang이 제안한 ResNet의 확장판으로 ResNet 블록에서는 합산을 통해 이전 layer와 현재 layer가 합쳐졌다. DenseNet의 경우, 연결을 통해 합쳐진다. 모든 layer를 이전 layer와 연결하고 현재 layer를 다음 layer에 연결한다. 이를 통해 더 매끄러운 기울기, 특징 변환 등과 같은 여러 가지 이점을 제공한다. 또한 parameter의 개수가 줄어든다.

### *MobileNet: Efficient Convolutional Neural Networks for Mobile Vision Application* | [arXiv](https://arxiv.org/abs/1704.04861)
MobileNet은 Google에서 연구한 Network로 version 1, 2, 3은 각 2017, 2018, 2019에 발표되었다. 고성능의 device가 아닌 vehicle, drone smart phone과 같은 환경에서는 computing power, memory가 부족하다. 따라서 battery performance가 중요한 곳을 목적으로 설계된 CNN이다. 작은 neural network를 만드는 방법에는 1. remove fully-connected layers으로 CNN parameters 90%를 FC layers가 차지한다. 2. kernel reduction으로 3x3을 1x1으로 변경해서 연산량을 줄인다. 3. channel reduction. 4. evenly spaced downsampling으로 초반에 downsampling을 많이 하면 accuracy가 떨어지지만 parameter가 적어지고 후반에 downsampling을 많이 하면 accuracy가 좋아지지만 parameter가 많아지기 때문에 적당히 사용하는 것이 좋다. 5. depthwise seperable convolutions. 6. shuffle operations. 7. distillation & compression.

기존의 CNN은 HxW 크기의 C개의 채널 image에 KxKxC 크기의 M개 filter를 곱하여 H'xW' 크기의 M 채널의 image를 생성한다. Depthwise & Pointwise convolution은 이와 달리 한 방향으로만 크기를 줄이는 전략이다. Depthwise convolution은 channel 개수는 줄어들지 않고 1개의 channel에서의 크기만 줄어든다. Pointwise convolution은 channel 숫자가 1개로 줄어든다. 기존 CNN의 parameter 수는 K^2CM, 계산량은 K^2CMH'W'이다. Depthwise convoltuion과 Pointwise convolution을 합한 parameter는 K^2C+CM, 계산량은 K^2CW'H' + CMW"H"이다. 만약 W'=W", H'=H"이면 W'H'C(K^2+M)이다. 즉, Depthwise convolution과 pointwise convolution을 합친 Separable convolutions의 계산량은 기존 CNN에 비해서 (1/M + 1/K^2)으로 K=3일 경우 약 8~9배의 효율을 보인다. (H=H'=H", W=W'=W"d 일 때)

## 🌳 Semantic Segmentation 🌳

### *R-CNN: Rich feature hierarchies for accurate object detection and semantic segmentation* | [arXiv](https://arxiv.org/abs/1311.2524)
2013년 UC Berkeley에서 발표한 object detection, semantic segmentation model이다. 이미지를 분류하는 것보다 이미지 안에 object인지 구분하는 것이 어려운 작업이다. R-CNN은 이를 위해 몇 단계를 거친다. 먼저 후보 이미지 영역을 찾아내는 region proposal/bounding box를 찾는 단계가 있다. Bounding box를 찾기 위해 색상이나 패턴 등이 비슷한 인접한 픽셀을 합치는 selective search 과정을 거친다. 다음 추출한 bounding box를 CNN의 입력으로 넣기 위해 강제로 사이즈를 통일 시킨다. 이 때 CNN은 훈련된 AlexNet의 변형된 버전이다. CNN의 마지막 단계에서 Support Vector Machine(SVM)을 사용하여 이미지를 분류한다. 그리고 최종적으로 분류된 object의 bounding box 좌표를 더 정확히 맞추기 위해 linear regression model을 사용한다.

### *Fast R-CNN* | [arXiv](https://arxiv.org/abs/1504.08083)
R-CNN의 문제점은 모든 bounding box에 대해 CNN, SVM, linear regression 3가지 모델을 훈련시켜야하기 떄문에 어렵다. 때문에 Fast R-CNN은 bounding box 사이에 겹치는 영역이 CNN을 통과시키는 것은 낭비라 생각했다. Region of Interset Pooling(RolPool)의 개념을 도입하여 selective search에서 찾은 bounding box 정보를 CNN을 통과시키면서 유지시키고 최종 CNN feature map으로부터 해당 영역을 추출하여 pooling한다. 이를 통해 bounding box마다 CNN을 거치는 시간을 단축시킨다. 또한 SVM 대신 CNN 뒤에 softmax를 놓고 linear regression 대신 softmax layer와 동일하게 뒤에 추가했다. Joint the feature extractor, classifier, regressor together in a unified framework.

### *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks* | [arXiv](https://arxiv.org/abs/1506.01497) | [PyCaffe](https://github.com/rbgirshick/py-faster-rcnn) | [PyTorch](https://github.com/longcw/faster_rcnn_pytorch) | [MatLab](https://github.com/ShaoqingRen/faster_rcnn)
Fast R-CNN에서 남은 bottleneck은 bounding box를 만드는 region proposal 단계이다. Faster R-CNN은 region proposal 단계를 CNN 안에 넣어서 문제를 해결했다. CNN을 통과한 feature map에서 sliding window를 이용해 각 anchor마다 가능한 bounding box의 좌표와 이 bounding box의 점수를 계산한다. 대부분 너무 홀쭉하거나 넓은 물체는 많지 않으므로 2:1, 1:1, 1:2 등의 몇가지 타입으로도 좋다. Faster R-CNN은 Microsoft에서 연구한 내용이다.

### *Mask R-CNN* | [arXiv](https://arxiv.org/abs/1703.06870) | [PyTorch](https://github.com/felixgwu/mask_rcnn_pytorch) | [TesforFlow](https://github.com/CharlesShang/FastMaskRCNN)
분할된 image를 masking하는 Mask R-CNN은 Facebook에서 연구한 내용으로, 각 픽셀이 object에 해당하는 것인지 아닌지를 masking하는 network를 추가했다. 이는 binary mask라 한다. 정확한 픽셀 위치를 추출하기 위해 CNN을 통과하면서 RolPool 영역에 위치에 생기는 소숫점 오차를 2D bilinear interpolation을 통해 감소시켰다. 이는 RolAlign이다. 

### *YOLO: You Only Look Once, Real-Time Object Detection* | [Homepage](https://pjreddie.com/darknet/yolo/)
YOLO는 이미지 내의 bounding box와 class probaility를 single regression problem으로 unified하여 이미지를 한번 보는 것으로 object의 종류와 위치를 추측한다. Single convolutional network를 통해 multiple bounding box에 대한 class probability를 계산하는 방식이다. 기존의 object detection method와 비교했을 때, YOLO의 상대적인 장점과 단점은 다음과 같다. 장점으로는 1. 간단한 처리과정으로 속도가 매우 빠르고 기존의 다른 real-time detection system들과 비교할 때 2배 정도 높은 mAP를 보인다. 2. Image 전체를 1번에 바라보는 방식으로 class에 대한 맥락적 이해도가 높아 낮은 background error(false-negative)를 보인다. 3. Object에 대한 좀 더 일반화된 특징을 학습하기 때문에 natural image로 학습하고 artwork에 테스트해도 다른 detection system에 비해 훨씬 높은 성능을 보인다. 단점으로는 상대적으로 정확도가 낮은데, 특히 작은 object일 수록이다.

Unified Detection은 input image를 S by S grid로 나눈다. 각각의 grid cell은 B개의 bounding box와 각 bounding box에 대한 confidence score를 갖는다. 만약 cell에 object가 없다면 confidence score는 0이 된다. 각각의 grid cell은 C개의 conditional class probability를 갖는다. 각각의 bounding box는 x, y, w, h, confidence로 구성된다. (x, y)는 bounding box의 중심점을 의미하며 grid cell의 범위에 대한 상대값이 입력된다. (w, h) 전체 image의 width, height에 대한 상대값이 입력된다. Test time에는 conditional class probability와 bounding box의 confidence score를 곱하여 class-specific confidence score를 얻는다. 논문에서는 YOLO의 성능평가를 위해 PASCAL VOC를 사용했으며 S, B, C에는 각각 7, 2, 20을 사용했다.

Network Design/Architecture은 GoogLeNet 모델의 24 convolutional layers and 2 fully connected layers을 기반으로 24 convolutional layers를 9개로 대체했다. 계산을 마치면 총 98개의 class specific confidence score를 얻게 되고, 이에 대해 각 20개의 class를 기준으로 non-maximum suppression을 하여 object에 대한 class 및 bounding box location을 결정한다. 

Loss function은 gird cell의 여러 bounding box 중 ground truth box와의 IOU가 가장 높은 bounding box를 predictor로 설정한다. object가 존재하는 grid cell i의 predictor bounding box j, object가 존재하지 않는 grid cell i의 bounding box j, object가 존재하는 grid cell i을 기호로 사용하고, ground truth box의 중심점이 어떤 grid cell 내부에 위치하면, 그 grid cell에는 object가 존재한다고 여긴다. [참고](https://curt-park.github.io/2017-03-26/yolo/)

YOLO의 한계는 1. 각 grid cell이 하나의 class만을 예측할 수 있으므로 작은 object 여러개가 있는 경우에는 제대로 예측하지 못한다. 2. bounding box의 형태가 training data를 통해서만 학습되므로 새로운 형태의 bounding box의 경우 정확히 예측하지 못한다. 3. 몇 단계의 layer를 거쳐서 나온 feature map을 대상으로 bounding box를 예측하므로 localization이 다소 부정확해지는 경우가 있다.

다른 real time object detection에 비해 높은 mAP를 보여주며 fast YOLO의 경우 가장 빠른 속도이다. Fast R-CNN과 비교하면 훨씬 적은 false positive이다. (low background error) Fast R-CNN과 같이 동작하면 보완하는 역할을 할 수 있다.

### *SSD: Single Shot MultiBox Detector* | [arXiv](https://arxiv.org/abs/1512.02325) 
SSD는 2015년에 UNC의 Wei Liu가 제안한 object detection method로, single deep neural network를 이용한다. **Multi-scale feature maps for detection**: 끝이 잘린 base network에 convolutional feature layers를 추가했다. 이 layers는 크기를 점차 줄여서 다양한 크기에서 prediction을 한다. Predicting detection을 하는 convolutional model은 feature layer들(Overfeat and YOLO)과 다르다. **Convolutional predictors for detection**

### *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks* | [arXiv](https://arxiv.org/abs/1905.11946) | [GitHub](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)

### *TecoGAN: Learning Temporal Coherence via Self-Supervision for GAN-based Video Generation* | [arXiv](https://arxiv.org/abs/1811.09393) | [GitHub](https://github.com/thunil/TecoGAN)

### *SinGAN: Learning a Generative Model from a Single Natural Image* | [arXiv](https://arxiv.org/pdf/1905.01164.pdf) | [GitHub](https://github.com/FriedRonaldo/SinGAN)
SinGAN은 InGan과 마찬가지로 a single natural image로 부터 여러 image를 생성하는 연구이지만, 차이점은 InGAN은 a single image에 대해서 여러 condition을 적용했지만, SinGAN은 unconditional한 방식이다.

### *InGAN: Capturing and Retargeting the "DNA" of a Natural Image* | [ICCV](https://openaccess.thecvf.com/content_ICCV_2019/papers/Shocher_InGAN_Capturing_and_Retargeting_the_DNA_of_a_Natural_Image_ICCV_2019_paper.pdf) | [Paper (arXiv)](https://arxiv.org/abs/1812.00231)

## 🌳 AutoML 🌳

### *NASNet, Learning Transferable Architectures for Scalable Image Recognition* | [arXiv](https://arxiv.org/abs/1707.07012)
2017년에 Google Brain의 Barret Zoph이 발표한 학습을 통해 modeal architecture를 찾는 network model이다. Reinforcement Learning(RL) search를 사용해서 architecture를 최적화하는 framework로 Neural Architecture Search(NAS) Net이다.

### *PNASNet, Progressive Neural Architecture Search* | [arXiv](https://arxiv.org/abs/1712.00559) | [PyTorch](https://github.com/chenxi116/PNASNet.pytorch)
2017년에 Google, JHU, Stanford에서 연구한 model로, model architecture를 학습해서 최적의 model architecture를 찾는 model이다.

### *ENASNet, Efficient Neural Architecture Search via Parameter Sharing* | [arXiv](https://arxiv.org/abs/1802.03268)
2018년에 Google, CMU, Stanford에서 연구한 model로, model architecture를 학습해서 최적의 model architecture를 찾는 model이다.

### *MnasNet, Platform-Aware Neural Architecture Search for Mobile* | [arXiv](https://arxiv.org/abs/1807.11626)
2018년에 Google에서 proposed model로, mobile environment에서 최적의 model architecture를 찾는 model이다. RL search를 이용한다.

### *YOSO, You Only Search Once: A Fast Automation Framework for Single-Stage DNN/Accelerator Co-design* | [arXiv](https://arxiv.org/abs/2005.07075)

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


#### Reference
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
- Selective Search for Object Recognition Slide, http://www.cs.cornell.edu/courses/cs7670/2014sp/slides/VisionSeminar14.pdf, 2021-03-09-Tue.
- Facebook Research Detectron GitHub, https://github.com/facebookresearch/Detectron, 2021-03-09-Tue.
- Facebook Research Detectron2 GitHub, https://github.com/facebookresearch/detectron2, 2021-03-09-Tue.
- OpenMMLab Detection Toolbox and Benchmark GitHub, https://github.com/open-mmlab/mmdetection, 2021-03-09-Tue.
- ResNeXt Blog KR, https://blog.airlab.re.kr/2019/08/resnext, 2021-03-10-Wed.
- ResNeXt Blog KR, https://everyday-deeplearning.tistory.com/entry/%EC%B4%88-%EA%B0%84%EB%8B%A8-%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0ResNext, 2021-03-10-Wed.
- SSD Blog KR, https://taeu.github.io/paper/deeplearning-paper-ssd/, 2021-03-15-Mon.
- SSD Blog KR, https://cocopambag.tistory.com/15, 2021-03-15-Mon.
- NASNet Blog KR, https://m.blog.naver.com/PostView.nhn?blogId=za_bc&logNo=221559482028&proxyReferer=https%3A%2F%2Fwww.google.com%2F, 2021-03-15-Mon.
- PNASNet Blog KR, https://m.blog.naver.com/PostView.nhn?blogId=za_bc&logNo=221576139392&proxyReferer=https:%2F%2Fwww.google.com%2F, 2021-03-16-Tue.
- PNASNet GitHub PyTorch, https://github.com/chenxi116/PNASNet.pytorch, 2021-03-16-Tue.
- ENASNet Blog KR, https://m.blog.naver.com/za_bc/221569478807, 2021-03-16-Tue.
- MnasNet Blog KR, https://developers-kr.googleblog.com/2018/09/mnasnet-towards-automating-design-of.html, 2021-03-15-Mon.
- MNasNet Blog KR, https://kmbin93.github.io/deeplearning/2020/07/21/MnasNet/, 2021-03-15-Mon.
- MNasNet Blog KR, https://m.blog.naver.com/PostView.nhn?blogId=za_bc&logNo=221570652712&proxyReferer=https:%2F%2Fwww.google.com%2F, 2021-03-15-Mon.
- mAP(mean Average Precision) for Object Detection Blog US, https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173, 2021-03-23-Tue.
