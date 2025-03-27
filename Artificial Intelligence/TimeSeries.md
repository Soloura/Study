# Time Series

## Forecasting

## Survey 

### [A Comprehensive Survey of Time Series Forecasting: Architectural Diversity and Open Challenges](https://arxiv.org/abs/2411.05793)

## Time Series Data

Time seies data is a sequence of data points collected or recorded at specific time intervals, where the order of data points is crucial. Examples include stock prices, weather data, sensor readings, and economic indicators. The key characteristic of time series data is the temporal dependency, meaning that past data points can influence future ones. Time series analysis involves understanding trends, seasonality, and patterns to forecast future values or detect anomalies.

## Methodologies

### Machine Learning Approaches for Time Series Data

1. Autoregressive Integrated Moving Average (ARIMA)

ARIMA is a classical time series model that combines autoregression, differencing, and moving averages to model time series data. It's effective for univariate time series forecasting, especially when the data is stationary (i.e., it has a constant mean and variance over time). However, ARIMA struggles with multivariate or nonlinear patterns in time series data.

2. Support Vector Regression (SVR) | [NIPS 1996](https://proceedings.neurips.cc/paper_files/paper/1996/file/d38901788c533e8286cb6400b40b386d-Paper.pdf)

SVR is a machine learning algorithm based on Support Vector Machines (SVM). It is commonly used for regression tasks, including time series forecasting. SVR can handle both linear and nonlinear relationships by applying kernel functions, making it versatile for complex time series patterns. However, it may not capture temporal dependencies as well as deep learning models.

3. Random Forest

Random Forest is an ensemble learning method that operates by constructing multiple decision trees. While primarily used for classification, it can be applied to time series regression tasks as well. It is robust to noise and can capture nonlinear patterns but lacks the capability to explicitly model the temporal aspect of time series data.

### Deep Learning Approaches for Time Series Data

1. Recurrent Neural Networks (RNN)

RNNs are designed to handle sequential data, making them a natural fit for time series analysis. In RNNs, the hidden state is updated at each time step, allowing the model to maintain information about previous inputs. However, standard RNNs suffer from vanishing gradients, which limits their ability to learn long-term dependencies in time series data.

2. Long Short-Term Memory (LSTM)

LSTM is a special type of RNN designed to address the vanishing gradient problem. It introduces memory cells that can maintain information for long periods, making LSTMs effective for learning long-term dependencies in time series data. LSTMs are widely used for time series forecasting, anomaly detection, and sequence modeling tasks.

3. Convolutional Neurla Network (CNN)

CNNs, typically used in image processnig, can also be applied to time series data by treading the time dimension as a spatial dimension. CNNs can capture local dependencies and patterns in the data, but they may not be as effective as RNNs or LSTMs for capturing long-term dependecies.

4. Attention Mechanism

The attentions mechanism allows models to focus on specific parts of the input sequence when making predictions. In time series data, attention helps models to prioritize important time steps or features, improving performance in tasks like forecasting or anomaly detection. Attention is particularly useful when different time points have varying levels of importance.

5. Transformer Models

Transformers, originally developed for natural language processing, have proven to be highly effective for time series data as well. The Transformer architecture relies entirely on self-attention mechanisms, making it capable of capturing long-range dependencies in the data. This ability has made Transformers popular in time series forecasting, with models such as Informer, LogTrans, and Temporal Fusion Transofmrer (TFT) being developed specifically for this domain.

---

### Reference
- Forecasting Economics and Financial Time Series: ARIMA vs. LSTM arXiv, https://arxiv.org/abs/1803.06386, 2024-09-23-Mon.
- Support Vector Regression Machines, https://proceedings.neurips.cc/paper_files/paper/1996/file/d38901788c533e8286cb6400b40b386d-Paper.pdf, 2024-09-23-Mon.
- Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting arXiv, https://arxiv.org/abs/2012.07436, 2024-09-23-Mon.
- Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting arXiv, https://arxiv.org/abs/1912.09363, 2024-09-23-Mon.
- Time Series Analysis, https://www.tableau.com/learn/articles/time-series-analysis, 2024-09-24-Tue.
- Time Series Analysis, https://www.timescale.com/blog/time-series-analysis-what-is-it-how-to-use-it/, 2024-09-24-Tue.
- Time Series Data Analysis, https://www.influxdata.com/what-is-time-series-data/, 2024-09-24-Tue.
- A Comprehensive Survey of Time Series Forecasting Architectural Diversity and Open Challenges arXiv, https://arxiv.org/abs/2411.05793, 2025-03-27-Thu.
