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

### XGBoost

XBoost는 Gradient Boosting Decision Tree (GBDT)의 확장으로, 대규모 데이터에 적합하게 설계된 확장 가능하고 효율적인 트리 기반 부스팅 기법이다. XGBoost을 통해 과적합을 방지하고, histogram-based learning, out-of-core computing 등 여러 최적화 기능을 지원하여 매우 큰 데이터셋에서도 빠르고 효과적으로 학습할 수 있다.

* 병렬 처리가 가능해 속도가 빠르며, 메모리 사용도 최적화됨
* 정규화를 통해 과적합 방지
* 다양한 파라미터 튜닝으로 유연성이 높음

### LightGBM

LightGBM은 XGBoost와 비슷한 Gradient Boosting Decision Tree (GBDT) 알고리즘이지만, 대용량 데이터에 더 적합하도록 설계되었다. 특히 Leaf-wise 성장 방식을 사용하여 학습 속도를 개선하고, Gradient-based One-Side Sampling (GOSS) 및 Exclusive Feature Bundling (EFB) 기술로 성능을 크게 향상시킨다. LightGBM은 특히 피처의 수가 많거나 대용량 데이터셋에서 뛰어한 성능을 발휘한다.

* 대규모 데이터에 적합 (효율적인 메모리 사용)
* 학습 속도가 매우 빠름
* Leaf-wise 트리 성장으로 정확도를 높임
