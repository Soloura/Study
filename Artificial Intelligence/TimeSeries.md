# Time Series

## Forecasting

## Survey 

### [A Comprehensive Survey of Time Series Forecasting: Architectural Diversity and Open Challenges](https://arxiv.org/abs/2411.05793)

* Fig.3: Evolution of Time Series Forecasting Models
  * Conventional Methods: Statistical, ML
  * Traditional DL Models: MLP, RNN, CNN, GNN
  * Transformers
  * Advanced Transformers: Patching, Cross-dimension, Exogenous, Additional Approaches
  * Advanced Traditional DL Models: MLP, CNN, RNN, GNN, Hybrid, Model-agnostic
  * Diffusion Models: Effective Conditional Embedding, Feature Extraction
  * Foundation Models: Sequential Modeling with LLMs, Pre-training
  * Mamba Models: Embedding & Multi-Scale, Sequetial Dependency, Channel Correlation, Efficient Modeling

* Table 1: Summary of Survey Papers on TSF

|ARTICLE|FEATURES|BROAD|FORECASTING|RECENT|REF|
|:------|:-------|:---:|:---------:|:----:|:-:|
|Time-series forecasting with deep learning: a survey|-|O|O|X|-|
|Forecast Methods for Time Series Data: A Survey|-|O|O|X|-|
|Deep Learning for Time Series Forecasting: Tutorial and Literature Survey|-|O|O|X|-|
|A Review on Deep Sequential Models for Forecasting Time Series Data|-|O|O|X|-|
|Transformers in Time Series: A Survey|-|X|O|X|-|
|Long Sequence Time-Series Forecasting with Deep Learning: A Survey|-|O|X|X|-|
|Machine Learning Advances for Time Series Forecasting|-|O|X|X|-|
|Diffusion Models for Time-Series Applications: A Survey|-|X|O|X|-|
|The Rise of Diffusion Models in Time-Series Forecasting|-|X|O|O|-|
|Foundation Models for Time Series Analysis: A Tutorial and Survey|-|X|O|O|-|
|Large Language Models for Time Series: A Survey|-|X|O|O|-|
|A Survey of Time Series Foundation Models|-|X|O|O|-|
|Mamba-360: Survey of State Space Models|-|X|O|O|-|
|THIS PAPER|-|O|O|O|-|

* Univariate Time Series Forecasting (UTSF): making predictions using only one variable. easy to understand, collection and management easier, limited information - external factors or interactions btw different variables.
* Multivariate Time Series Forecasting (MTSF): making predictions using multiple variables simultaneously. capture complex relationships, higher accuracy - models complex, require more data, chanllenging to handle, rick of overfitting.
* Short-Term Time Series Forecasting (STSF): Short-term time series forecasting focuses on predicitons for the near future, making it suitable for tasks that require quick responses. simple, easy to train and implement, limits its applicability
* Long-Term Time Series Forecasting (LTSF): Long-term time series forecasting deals with predictions for the distant future, with forecast horizons increasingly extending to several months, years, or beyond. It is valuable for long-term strategy planning, investment decisions, and policymaking, addressing many real-world problems. By identifying long-term trends and cycles, organizations can prepare accordingly, highlighting its significance. However, predicting the distant future is
challenging, and extensive research is being conducted to improve accuracy.

* Properties of Time Series Data
  * Tempora Order
  * Autocorrelation
  * Trend
  * Seasonality
  * Outliers or noise
  * Irregularity
  * Cycles
  * Non-stationarity

* Properties of Multivariate Time Series Data
  * Interdependency: In time series data collected simultaneously across multiple channels, these variables can be correlated. Understanding the interactions between variables is important as it can help comprehend complex patterns in time series data
  * Exogenous Variables: External factors or variables influence time series data. Although not included in the data itself, these variables can provide latent information, and considering them during modeling can lead to significant performance improvements
  * Contextual Information: Specific events, such as policy changes or natural disasters, occurring at the time of observation can affect time series data and create complex patterns
 
* Time Series Forecasting Datasets

|DATASET|CHANNEL|LENGTH|DESC|FREQ|SRC|
|:------|------:|-----:|:---|---:|---|
|ETTm1, ETTm2|7|69680|-|15 mins|-|
|ETTh1, ETTh2|7|17420|-|Hourly|-|
|Electricity|321|26304|-|Hourly|-|
|Traffic|862|17544|-|Hourly, Weekly|-|
|Weather|21|52696|-|10 mins|-|
|Exchange|8|7588|-|Daily|-|
|ILI|7|966|-|Weekly|-|
...

* 2.4. Evaluation Metrics
  * Metrics for Deterministic Models
    * Mean Absolute Error (MAE): measures the average of the absolute differences between predicted and actual values.
    * Mean Squared Error (MSE): calculates the average of the squared differences between predicted and actual values.
    * Root Mean Squared Error (RMSE): the square root of the mean squared error, which is more sensitive to large errors.
  * Relative Error Metrics
    * Mean Absolute Percentage Error (MAPE): evaluates the relative error by calculating the average of the absolute percentage differences between predicted and actual values.
    * Symmetric Mean Absolute Percentage Error (sMAPE): a modified version of MAPE that addresses its asymmetry by taking the average of the absolute percentage differences in a symmetric way.
  * Metrics for Probabilistic Models
    * Continuous Ranked Probability Score (CRPS): measures the difference between the predicted probability distribution and the actual distribution, providing a comprehensive evaluation of probabilistic forecasts.
  * Other Metrics
    * Coefficient of Determination (R**, R^2): indicates how well the model explains the variabiity of the data.
    * Mean Forecast Error (MFE): represents the bias in predictions by calculating the average of the forecast errors.
    * Cumulative Forecast Error (CFE): the sum of all forecast errors over the forecast horizon, which can indicate the trend of prediction errors over time.
   
* 3. Historical TSF Models
  * 3.1. Conventional Methods (Before Deep Learning)
    * 3.1.1. Statistical Models
    * 3.1.2. Machine Learning Models
  * 3.2. Traditional Deep Learning Models
    * 3.2.1. MLPs: The Emergence and Constraints of Early Artificial Neural Networks
    * 3.2.2. RNNs: The first neural network capable of processing sequential data and modeling temporal dependencies
    * 3.2.3. CNNs: Extracting key patterns in time series data beyond just images
    * 3.2.4. GNNs: Structurally modeling relationships between variables
  * 3.3 The Prominence of Transformer-based Models
    * 3.3.1. Transformer Variants
    * 3.3.2. Limitation of Transformer-based Models
  * 3.4 Uprising of Non-Transformer-based Models

* 4. New Exploration of TSF Models
  * 4.1. Overcoming Limitations of Transformer
    * 4.1.1. Patching Technique
    * 4.1.2. Cross-Dimension
    * 4.1.3. Exogenous Variable
    * 4.1.4. Additional Approaches
  * 4.2. Growth of Traditional Deep Learning Models
    * 4.2.1. MLP-Based Models
    * 4.2.2. CNN-Based Models
    * 4.2.3. RNN-Based Models
    * 4.2.4. GNN-Based Models
    * 4.2.5. Hybrid Models
    * 4.2.6. Model-Agnostic Frameworks
  * 4.3. Emergence of Foundation Models
    * 4.3.1. Sequence Modeling with LLMs
    * 4.3.2. Pre-training
  * 4.4. Advance of Diffusion Models
    * 4.4.1. Effective Conditional Embedding
    * 4.4.2. Time Series Feature Extraction
    * 4.4.3. Additional Approaches
  * 4.5. Debut of the Mamba
    * 4.5.1. History of the SSM(State Space Model)
    * 4.5.2. Introduction of the Mamba
    * 4.5.3. Applications of the Mamba

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
