# :hammer_and_wrench: [Optimization](https://en.wikipedia.org/wiki/Mathematical_optimization)

### A Survey of Optimization Methods from a Machine Learning | [arXiv, 2019](https://arxiv.org/pdf/1906.06821.pdf)

### Gradient Descent (GD)

The gradient descent method is the earliest and most common optimization method.

The idea of the gradient descent method is that variables update iteratively in the (oppostie) direction of the gradients of the objective function.

The update is performed to gradually converge to the optimal value of the objective function.

The learning rate delta determines the step size in each iteration, and thus influences the number of iterations to reach the optimal value.

:key: Properties:
- Solve the optimal value along the direction of the gradient descent.
- The method converges at a linear rate.

Advantages:
- Solution is global optimal when the objective function is convex.

Disadvantages:
- In each parameter update, gradients of total samples need to be calculated, so the calculation cost is high.

### Stochastic Gradient Descent (SGD) | A Stochastic Approximation Method | [Ann. Math. Statist., 1951](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-22/issue-3/A-Stochastic-Approximation-Method/10.1214/aoms/1177729586.full)

:key: Properties:
- The update parameters are calculated using a randomly sampled mini-batch.
- The method converges at a sublinear rate.

Advantages:
- The calculation time for each update does not depend on the total number of training samples, and a lot of calculation cost is saved.

Disadvantages:
- It is diffcult to choose an appropriate learning rate, and using the same learning rate for all parameters is not appropriate.
- The solution may be trapped at the saddle point in some cases.

### AdaGrad | Adaptive Subgradient Methods for Online Learning and Stochastic Optimization | [JMLR, 2011](https://jmlr.org/papers/v12/duchi11a.html)

:key: Properties:
- The learning rate is adaptively adjusted according to the sum of the squares of all historical gradients.

Advantages:
- In the early stage of training, the cumulative gradient is smaller, the learning rate is larger, and learning speed is faster.
- The method is suitable for dealing with sparse gradient problems.
- The learning rate of each parameter adjusts adaptively.

Disadvantages:
- As the training time increases, the accumulated gradient will become larger and larger, making the learning rate tend to zero, resulting in ineffective parameter updates.
- A manual learning rate is still needed.
- It is not suitable for dealing with non-convex problems.

### Adam | Adam: A method for stochastic optimization | [ICLR, 2015](https://dblp.org/rec/journals/corr/KingmaB14.html) | [arXiv, 2017 v9](https://arxiv.org/abs/1412.6980)

:key: Properties:
- Combine the adaptive methods and the momentum method. 
- Use the first-order moment estimation and the second-order moment estimation of the gradient to dynamically adjust the learning rate of each parameter. 
- Add the bias correction.

Advantages:
- The gradient descent process is relatively stable.
- It is suitable for most non-convex optimization problems with large data sets and high dimentsional space.

Disadvantages:
- The method may not converge in some cases.

## :hammer_and_wrench: [`tf.keras.optimizers`](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)

---

### Reference
- Mathematical optimization, https://en.wikipedia.org/wiki/Mathematical_optimization, 2023-01-23-Mon.
- Module: tf.keras.optimizers, tps://www.tensorflow.org/api_docs/python/tf/keras/optimizers, 2023-01-23-Mon.
- A Survey of Optimization Methods from a Machine Learning, https://arxiv.org/pdf/1906.06821.pdf, 2023-01-23-Mon.
- A Stochastic Approximation Method, https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-22/issue-3/A-Stochastic-Approximation-Method/10.1214/aoms/1177729586.full, 2023-01-23-Mon.
- Adaptive Subgradient Methods for Online Learning and Stochastic Optimization, https://jmlr.org/papers/v12/duchi11a.html, 2023-01-23-Mon.
- Adam: A Method for Stochastic Optimization, https://dblp.org/rec/journals/corr/KingmaB14.html, 2023-01-23-Mon.
- Adam: A Method for Stochastic Optimization, https://arxiv.org/abs/1412.6980, 2023-01-23-Mon.
