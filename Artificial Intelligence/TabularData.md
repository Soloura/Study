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

* VIME (Variational Information Maximization Embedding): 데이터의 비지도 학습을 통해 학습 
