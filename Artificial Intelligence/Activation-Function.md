# :hammer_and_wrench: [Activation Function](https://en.wikipedia.org/wiki/Activation_function)

`This page is from the 'Computer Vision' page.`

The activation function of a node defines the output of that node given an input or set of inputs.

A standard integrated circuit can be seen as a digital network of activation functions that can be ON/1 of OFF/0, depending on input. This is similar to the linear perceptron in neural networks.

However, only nonlinear activation functions allow such networks to compute nontrivial problems using only a small number of nodes, and such activation functions are called nonlinearities.

### [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) | [Blog (KR)](http://taewan.kim/post/sigmoid_diff/)

S(x) = 1/(1+e^(-x)) = e^x/(e^x+1) = 1 - S(-x)

A sigmoid function is a mathematical function having a characteristic "S"-shaped curve or sigmoid curve.

Properties:
- It is monotomic.
- It has a first derivative which is bell shaped.
- Integral of any continuous, non-negative, bell-shaped function (with one local maximum and no local minimum, unless degenerate) will be sigmoid.

```Python
def sigmoid(x): return 1 / (1 + exp(-1))
```

```Python
def diff_sigmoid(x): return sigmoid(x) * (1 - sigmoid(x))
```

### ReLU

```Python
def relu(x): return max(0, x)
```

### ReLU6 | [Blog (KR)](https://gaussian37.github.io/dl-concept-relu6/)

```Python
def relu6(x): return min(6, relu(x))
```

### Swish | [Blog (KR)](https://eehoeskrap.tistory.com/440)

Swish는 깊은 neural network에서 ReLU보다 높은 accuracy를 가진다.

모든 batch size에 대해 Swish는 ReLU에 비해 accuracy가 높다.

Bounded below, unbounded above 특성이 있다.

Non-monotonic 함수이며 1차, 2차 미분을 갖는다.

```Python
def swish(x): return x * tf.nn.sigmoid(x) # x * torch.sigmoid(x)
```

### Hard Swish

```Python
def hard_swish(x): return x * (relu6(x+3))/6
```

### Mish | [Blog (KR)](https://eehoeskrap.tistory.com/440)

Mish는 무한대로 가기 때문에(unbounded above) 캡핑으로 인한 포화를 피할 수 있다.

Bounded below이기 때문에 strong regularation이 나타날 수 있고 overfitting을 감소시킬 수 있다.

또한, 약간의 음수를 허용하기 때문에 ReLU zero bound보다는 gradient가 더 잘 흐를 수 있다.

범위는 -0.31 ~ infinite이다.

Non-monotonic 함수이며 1차, 2차 미분을 갖는다.

```Python
def mish(x): return x * tf.nn.tanh(tf.nn.softplus(x)) # x * torch.tanh(F.softplus(x))
```

### Universal activation function for machine learning | [Scientific reports, 2021](https://www.nature.com/articles/s41598-021-96723-8)

Propose a universal activation function (UAF) that achieves near optimal performance in quantification, classification, and reinforcement learning problems.

## :hammer_and_wrench: [`tf.keras.layers.Activation`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Activation)

--- 

### Reference
- Activation function, https://en.wikipedia.org/wiki/Activation_function, 2023-01-23-Mon.
- Sigmoid function, https://en.wikipedia.org/wiki/Sigmoid_function, 2023-01-23-Mon.
- Swish vs. Wish Blog KR, https://eehoeskrap.tistory.com/440, 2021-12-01-Wed.
- ReLU6 Blog KR, https://gaussian37.github.io/dl-concept-relu6/, 2021-12-02-Thu.
- Sigmoid Blog KR, http://taewan.kim/post/sigmoid_diff/, 2021-12-02-Thu.
- Universal activation function for machine learning, https://www.nature.com/articles/s41598-021-96723-8, 2023-01-23-Mon.
- tf.keras.layers.Activation, https://www.tensorflow.org/api_docs/python/tf/keras/layers/Activation, 2023-01-23-Mon.
