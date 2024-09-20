# Tabular Data

* consists of row and columns as datasheets and table of databases
* row is data instance or record
* column is a property or feature
* structured data, fixed data type, query

### Machnine Learning

* Linear Regression: 연속적인 값을 예측, 피처의 독립성과 선형성이 있을 때 효과적
* Logistic Regression: 이진 분류 문제에 많이 사용됨
* Decision Trees:데이터를 트리 구조로 분류, 해석이 쉽고 비선형 관계를 잘 다룸
* Random Forests: 여러 DTs를 앙상블로 사용하는 방법, 강력하고 안정적이며 overfitting을 잘 방지함
* Gradient Boosting Machine (GBM): 모델의 오차를 점진적으로 줄여가는 앙상블 기법, XGBoost, LightGBM 성능 뛰어남

### Ensemble Methods

* Baggings: 여러 모델을 병렬로 학습한 후 경과를 결합하여 최종 예측을 만드는 기법, 랜덤 포레스트
* Boosting: 각 모델이 직렬로 학습하여 이전 모델의 오차를 보정하는 방식, XGBoost, LightGBM, CatBoost

### Deep Learning

* TabNet: tabular data에 특화된 모델, 학습 가능한 모델 해석을 제공, 피처 선택과 모델 해석이 가능함
* FT-Transformer: tabular 데이터를 transformer 구조로 처리하는 모델

### Self-Supervised Learning and Representation Learning

* VIME (Variational Information Maximization Embedding): 데이터의 비지도 학습을 통해 학습을 통해 학습 성능을 높임

##

### XGBoost | [arXiv 2016](https://arxiv.org/abs/1603.02754)

XBoost는 Gradient Boosting Decision Tree (GBDT)의 확장으로, 대규모 데이터에 적합하게 설계된 확장 가능하고 효율적인 트리 기반 부스팅 기법이다. XGBoost을 통해 과적합을 방지하고, histogram-based learning, out-of-core computing 등 여러 최적화 기능을 지원하여 매우 큰 데이터셋에서도 빠르고 효과적으로 학습할 수 있다.

* 병렬 처리가 가능해 속도가 빠르며, 메모리 사용도 최적화됨
* 정규화를 통해 과적합 방지
* 다양한 파라미터 튜닝으로 유연성이 높음

### LightGBM | [NIPS 2017](https://papers.nips.cc/paper_files/paper/2022/hash/77911ed9e6e864ca1a3d165b2c3cb258-Abstract.html)

LightGBM은 XGBoost와 비슷한 Gradient Boosting Decision Tree (GBDT) 알고리즘이지만, 대용량 데이터에 더 적합하도록 설계되었다. 특히 Leaf-wise 성장 방식을 사용하여 학습 속도를 개선하고, Gradient-based One-Side Sampling (GOSS) 및 Exclusive Feature Bundling (EFB) 기술로 성능을 크게 향상시킨다. LightGBM은 특히 피처의 수가 많거나 대용량 데이터셋에서 뛰어한 성능을 발휘한다.

* 대규모 데이터에 적합 (효율적인 메모리 사용)
* 학습 속도가 매우 빠름
* Leaf-wise 트리 성장으로 정확도를 높임

### CatBoost | [NIPS 2017](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/http://learningsys.org/nips17/assets/papers/paper_11.pdf)

범주형 데이터 처리를 매주 효과적으로 하는 부스팅 알고리즘이다. 기존의 부스팅 알고리즘들이 범주형 데이터를 처리할 때 발생하는 편향 문제를 해결하기 위해 설계되었다. Target encoding을 자동으로 처리하고, Ordered boosting을 도입해 데이터 누수를 방지한다. 이를 통해 범주형 피처가 많은 데이터에서도 매우 우수한 성능을 발휘한다.

* 범주형 피처 처리에 특화된 알고리즘
* 데이터 누수를 방지하는 독자적인 boosting 방식
* 기본 설정으로도 좋은 성능을 낼 수 있음

### TabNet | [arXiv 2020](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/1908.07442)

TabNet은 tabular 데이터를 처리하는 딥러닝 모델로, tabular 데이터의 특성을 반영한 sparse attention 메커니즘을 사용한다. 이 모델은 입력 피처 중 일부만 선택해 사용하는 피처 선택 (feature selection) 기능을 통해, 학습 효율성을 높이고 모델의 해석 가능성을 제공한다. TabNet은 전통적인 머신러닝 기법들, 특히 부스팅 기법과 비교했을 때도 경쟁력있는 성능을 보이며, 피처 중요도를 직접 확인할 수 있어 해석 가능성이 높다.

* sparse attention을 통한 피처 선택 가능
* 학습 가능한 피처 중요도 제공
* tabular 데이터에서 경쟁력 있는 성능

### FT-Transformer | [arXiv 2023](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/2106.11959)

FT-Transformer는 Transformer 모델을 tabular 데이터에 적용한 방법론이다. 전통적인 트리 기반 모델과 딥러닝 모델의 차이를 좁히기 위해, 이 모델은 self-attention 메커니즘을 사용하여 피처 간의 복잡한 관계를 모델링한다. 이 방법은 tabular 데이터에서 딥러닝의 활용 가능성을 넓히고, 기존의 트리 기반 모델과 딥러닝 모델 간의 성능 격차를 줄이는 중요한 연구이다.

* self-attention 메커니즘을 활용해 피처 간 상호작용을 학습
* tabular 데이터에서 딥러닝의 적용을 시도

### VIME

VIME은 self-supervised learning과 semi-supervised learning을 tabular 데이터에 적용한 모델이다. VIME는 학습 데이터가 부족한 환경에서 매우 유용하며, unlabeled 데이터로부터 유용한 피처 표현을 학습할 수 있도록 한다. 이를 통해 라벨이 제한적인 환경에서도 tabular 데이터의 성능을 크게 향상시킬 수 있다.

* labeled 데이터가 부족한 상황에서 성능 향상
* self-supervised learning을 tabular 데이터에 적용
