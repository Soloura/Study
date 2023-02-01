# :hammer_and_wrench: Loss Function | [Google](https://developers.google.com/machine-learning/crash-course/descending-into-ml/training-and-loss)

Loss is the penalty for a bad prediction. That is, loss is a number indicating how bad the model's prediction was on a single example. If the model's prediction is perfect, the loss is zero; otherwise, the loss is greater. The goal of training a model is to find a set of weights and biases that have low loss, on average, across all examples.

## :chart: Regression

### Mean Absolute Error (MAE)

### Mean Square Error (MSE)

### Root Mean Square Error (RMSE)

## :bow_and_arrow: Classification

### Entropy

Entropy is an average amount of information about probabilistically occurring events. The amount of information is defined as follows and can be seen as representing the degree of surprise.

The more difficult the prediction, the greater the amount of information and the greater the entropy.

### Binary Cross Entropy

### Categorical Cross Entropy

## :hammer_and_wrench: [`tf.keras.losses`](https://www.tensorflow.org/api_docs/python/tf/keras/losses)

---

### Reference
- Loss Function, https://brunch.co.kr/@mnc/9, 2023-01-22-Sun.
- Descending into ML: Training and Loss, https://developers.google.com/machine-learning/crash-course/descending-into-ml/training-and-loss, 2023-01-23-Mon.
- Loss function, https://en.wikipedia.org/wiki/Loss_function, 2023-01-23-Mon.
- Module: tf.keras.losses, https://www.tensorflow.org/api_docs/python/tf/keras/losses, 2023-01-23-Mon.
- [손실함수] Binary Cross Entropy, https://curt-park.github.io/2018-09-19/loss-cross-entropy/, 2023-02-01-Wed.
