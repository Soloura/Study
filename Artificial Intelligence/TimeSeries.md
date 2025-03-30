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

* Summary of Survey Papers on TSF > Table 1: Summary of Survey Papers on Time Series Forecasting

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

* Properties of Time Series Data > Fig. 5: Properties of Time Series Data
  * Tempora Order
  * Autocorrelation
  * Trend
  * Seasonality
  * Outliers or noise
  * Irregularity
  * Cycles
  * Non-stationarity

* Properties of Multivariate Time Series Data > Fig. 6: Properties of Multivariate Time Series Data
  * Interdependency: In time series data collected simultaneously across multiple channels, these variables can be correlated. Understanding the interactions between variables is important as it can help comprehend complex patterns in time series data
  * Exogenous Variables: External factors or variables influence time series data. Although not included in the data itself, these variables can provide latent information, and considering them during modeling can lead to significant performance improvements
  * Contextual Information: Specific events, such as policy changes or natural disasters, occurring at the time of observation can affect time series data and create complex patterns
 
* Time Series Forecasting Datasets > Table 2: Datasets Frequently Used in Time Series Forecasting Models

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

#### 2.4. Evaluation Metrics > Table 4: Evaluation Metrics for Time Series Forecasting
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

#### 3. Historical TSF Models > Fig. 8: Remarkable Historical TSF Models

* 3.1. Conventional Methods (Before Deep Learning)
  * 3.1.1. Statistical Models
    * Autoregressive Integrated Moving Average (ARIMA): AR uses a linear combination of past values to predict the current value. MA employs a linear combination of past error terms to predict the current value. I removes non-stationarity by differencing the data to achieve stationarity. SARIMA: incorporate seasonal differencing.
  * 3.1.2. Machine Learning Models
    * Decision Trees, Classification and Regression Tree (CART), Support Vector Machine (SVM), Support Vector Regression (SVR), Gradient Boosting Machines (GBM), XGBOost.
* 3.2. Traditional Deep Learning Models
  * 3.2.1. MLPs: The Emergence and Constraints of Early Artificial Neural Networks
    * Multi-layer Perceptron (MLP).
  * 3.2.2. RNNs: The first neural network capable of processing sequential data and modeling temporal dependencies
    * Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU), Dilated RNN, DA-RNN, MQ-RNN.
  * 3.2.3. CNNs: Extracting key patterns in time series data beyond just images (Convolutional Neural Networks)
    * Neocognitron, LeNet, WaveNet, Temporal Convolutional Networks (TCNs), DeepAR, DCRNN, TPA-LSTM.
  * 3.2.4. GNNs: Structurally modeling relationships between variables
    * Graph Neural Networks (GNNs), Graph Convolutional Networks (GCNs), Spatial Temporal Graph Convolutional Networks (ST-GCN), Graph Attention Networks (GATs), Dynamic Graph Neural Networks (DyGNNs), Temporal Graph Networks (TGNs), Logsparse Transformer.
* 3.3 The Prominence of Transformer-based Models
  * 3.3.1. Transformer Variants
    * LogTrans, Reformer, Informer, Autoformer, Pyraformer, Fedformer, Non-stationary Transformer.
  * 3.3.2. Limitation of Transformer-based Models
    * Efficiency Chanllenges - quadratic scaling in computational and memory complexity with window length have not fully resolved the issues than MLP-based or convolutional models with O(L) complexity.
    * Finite Context Window - unlike models RNNs or State Space Models (SSMs), fundamental limitation in modeling data beyond a finite window.
    * Ineffectiveness of Expanding Input Window Length - minimal or no performance improvement observed when increasing the input window length. Transformer-based models tend to overfit noise rather than extract long-term temporal information when provided with longer sequences.
* 3.4 Uprising of Non-Transformer-based Models
  * Limitations when dealing with long-term time series data.
    * The point-wise operations of the self-attention mechanism have quadratic time complexity.
    * Storing the relationship information for all input token pairs requires substantial memory usage, applying Transformers in environments with limited GPU memory becomes challenging.
    * Length of the look-back window exceeds the structural capacity, learning long-term dependencies becomes challenging.
    * Due to the high complexity of the model, large-scale and high-quality datasets are required. In the absence of sufficient data, can occur leading to a drop in model performance - sensitive.

#### 4. New Exploration of TSF Models

* 4.1. Overcoming Limitations of Transformer > Table 5: Taxonomy and Methodologies of Transformer Models for Time Series Forecasting - LTSF-Linear > traditional transformer-based models. Transformers still demonstrate superior performance in handling long-term dependencies in sequential data compared to other models.
  * 4.1.1. Patching Technique; dividing input sequences into multiple patches - preserves the information within each patches, thereby enhancing locality (instead of individual points), processes fewer tokens: PatchTST, MTST, PETformer.
  * 4.1.2. Cross-Dimension: Crossformer, DSformer, CARD, iTransformer, VCformer, GridTST, UniTST, DeformTime.
  * 4.1.3. Exogenous Variable: TimeXer, TGTSF.
  * 4.1.4. Additional Approaches
    * Generalization; model generalization, avoid overfitting, and achieve consistent performance across diverse datasets: SAMformer, Minusformer.
    * Multi-scale; extracts more information from time series data across various scales, offering distinct advantages: Scaleformer, Pathformer.
    * Decoder-only; as LLaMA3, simpler and involves less complex computations, resulting in faster training and inference. Avoid the temporal information loss often associated with the self-attention mechanism in encoders: CATS.
    * Feature Enhancement: Fredformer, Basisformer.
* 4.2. Growth of Traditional Deep Learning Models > Table 6: Comparison of Other Deep Learnig Models with Transformers in Terms of Criteria & Table 7: Taxonomy and Methodologies of Traditional Deep Learning Architectures for Time Series Forecasting
  * 4.2.1. MLP-Based Models: Koopa, TSMixer, FreTS, TSP, FITS,* U-Mixer, TTMs, TimeMixer, CATS, HDMixer, SOFTS, SparseTSF, TEFN, PDMLP, AMD.
  * 4.2.2. CNN-Based Models: TimesNet, PatchMixer, ModernTCN, ConvTimeNet, ACNet, FTMixer.
  * 4.2.3. RNN-Based Models: PA-RNN, WITRAN, SutraNets, CrossWaveNet, DAN, RWKV-TS, CONTIME.
  * 4.2.4. GNN-Based Models: MSGNet, TMP-Nets, HD-TTS, ForecastGrapher.
  * 4.2.5. Hybrid Models: WaveForM, TSLANet, DERITS, BiTGraph.
  * 4.2.6. Model-Agnostic Frameworks: RobustTSF, PDLS, Leddam, InfoTime, CCM, HCAN, TDT Loss.
* 4.3. Emergence of Foundation Models > Table 8: Taxonomy and Methodologies of Foundation Models for Time Series Forecasting
  * 4.3.1. Sequence Modeling with LLMs: GPT4TS, PromptCast, LLMTime, Time-LLM.
  * 4.3.2. Pre-training: Lag-LLaMA, TimesFM, CHRONOS, Uni2TS.
* 4.4. Advance of Diffusion Models > Table 9: Taxonomy and Methodologies of Diffusion Models for Time Series Forecasting
  * 4.4.1. Effective Conditional Embedding: TimeGrad, CSDI, SSSD, TimeDiff, TMDM.
  * 4.4.2. Time Series Feature Extraction
    * Decomposition: Diffusion-TS.
    * Frequency Domain
    * Multi-Scale: MG-TSD, mr-Diff.
  * 4.4.3. Additional Approaches
    * Score-Based Generative Modeling through Stochastic Differential Equations (SDEs): SDEs, DDPM, ScoreGrad, D^3M.
    * Latent Diffusion Model: Latent Diffusion Model, LDT.
    * Guidance: Diffusion-TS, LDT, TSDiff.
* 4.5. Debut of the Mamba
  * 4.5.1. History of the SSM(State Space Model): RNNs lost their dominance by Transformers due to the limitations in information encapsulation of a context (single vector) in RNN-based encoder-decoder models and the slow training speed. The parallelism of the attention mechanism and its ability to focus on all individial pieces of information overcame the limitations of RNNs and demonstrated superior performance. Computation complexity, limits the window length, increased memory requirements for processing long sequences. 
  * 4.5.2. Introduction of the Mamba
  * 4.5.3. Applications of the Mamba > Table 10: Taxonomy and Methodologies of Mamba Models for Time Series Forecasting
    * Embedding and Multi-Scale Learning
      * TimeMachine
    * Channel Correlation Learning
      * S-Mamba
      * SiMBA
      * MambaTS
      * C-Mamba
    * Sequence Information and Dependency Learning
      * Mambaformer
      * Bi-Mamba+
      * DTMamba
    * Theoretical Frameworks and Efficient Modeling
      * Time-SSM
      * Chimera

#### Table 5: Taxonomy and Methodologies of Transformer Models for Time Series Forecasting

|Main Improvement|Model Name|Main Methodology|Channel Correlation|Enc/Dec|Pub|
|:--------------:|:--------:|:---------------|:-----------------:|:-----:|--:|
|Patching Technique|PatchTST|Patching, Channel Independence|CI|Enc|2023|
|Patching Technique|MTST|Multiple Patch-based Tokenizations|CI|Enc|2024|
|Patching Technique|PETformer|Placeholder-enhanced Technique|CI|Enc|2022|
|Cross-Dimension|Crossformer|Dual Attention: Cross-time & Cross-dimension|CD|Enc & Dec|2023|
|Cross-Dimension|DSformer|Dual Sampling & Dual Attention|CD|Enc|2023|
|Cross-Dimension|CARD|Dual Attention, Token Blend Module for multi-scale|CD|Enc|2024|
|Cross-Dimension|iTransformer|Attention on Inverted Dimension|CD|Enc|2024
|Cross-Dimension|VCformer|Variable Correlation Attention Considering Time Lag, Koopman Temporal Detector for Non-stationarity|CD|Enc|2024|
|Cross-Dimension|GridTST|Dual Attention with original Transformer|CD|Enc|2024|
|Cross-Dimension|UniTST|Unified Attention by Flattening|CD|Enc|2024|
|Cross-Dimension|DeformTime|Deformable Attention Blocks|CD|Enc|2024|
|Exogenous Variable|TimeXer|Integration of Endogenous and Exogenous Information|CD|Enc|2024|
|Exogenous Variable|TGTSF|Exogenous Variable with Description, News|CD|Enc & Dec|2024|
|Generalization|SAMformer|SAM(sharpness-aware minimization|CD|Enc|2024|
|Generaliation|Minusformer|Dual-stream and subtraction mechanism|CD|Enc|2024|
|Multi-scale|Scaleformer|Multi-scale framework|Any|Enc & Dec|2023|
|Multi-scale|Pathformer|Adaptive Multi-scale Blocks|CD|Enc & Dec|2024|
|Decoder-only|CATS|Cross-Attention-ONly Transformer|CI|Dec|2024|
|Feature Enchancement|Fredformer|Frequency Debias|CD|Enc|2024|
|Feature Enchancement|BasisForemr|Automatic Learning of a Self-adjusting Basis|CD|Dec|2023|

#### Table 6: Comparison of Other Deep Learning Models with Transformers in Terms of Criteria

|Criteria|Transformer-based models|MLP-based models|CNN-based models|RNN-based models|GNN/GCN-based models|
|:------:|:----------------------:|:--------------:|:--------------:|:--------------:|:------------------:|
|Structure|Complex self-attention mechanism|Simple layer, relatively easy to implement and interpret|Utilized convolutional layers, effectively captureing specific local patterns|Specialized in sequential data processing, effectively handling temporal dependencies but may struggle with long sequences|Learns relationships between nodes using graph structures, effectively capturing complex relationships|
|Data Requirements|Requires large datasets|Can train on smaller datasets|Can train on smaller datasets|Can train on smaller datasets, suitable for quick training with limited data|Can achieve high performance with relatively small datasets|
|Training Time|Relatively slow due to global attention mechanisms|Relatively fast|Generally faster due to localized computations|Trains effectively on smaller datasets but can be slow for long sequences|Varies depending on graph complexity|
|Model Size|Comparatively larger and more parameter-intensive|Comparatively small, efficient use of resources|Typically smaller and more parameter-efficient, making it resource-efficient and scalable|Comparatively small|Depends on graph size and complexity, achieveing high performance with fewer parameters in specific problems|
|Interpretability|Difficult to interpret|Relatively high interpretability|More interpretable through visualizations of filters and feature maps|Moderately interpretable, easier to understand and explain model behavior|Moderately interpretable depending on graph structure and model complexity|
|Performance|Suitable for learning long-term dependencies|Suitable for short-term predictions with sufficient performance in many cases|Excels at capturing local temporal dependencies, superior for problems with strong local patterns|Suitable for short-term and sequential dependencies, providing sufficient performance in specific time series problems|Excels at learning complex dependencies between nodes, offering high performance in learning specific relationship patterns|
|Flexibility|Requires complex adjustments|Easily adjustable for specific problems|Easily customizable with various types of convolutions|Extensible with various RNN architectures|Can handle various graph structures and formats, adaptable to different data types and structures|
|Application|Suitable for complex time series problems or NLP-related tasks|Versatile for various general forecasting problems|Well-suitable for applications requiring spatial and temporal locality, effective for a wide range of time series problems|Effective for sequential data and time series forecasting, but struggles with long-term dependencies without modifications|Suitable for complex graph structures in tasks like social networks, recommendations systems, and time series graphs|
|Hardware Requirements|High due to their complex structure and computationally intensive self-attention mechanisms|Lower due to their simpler structure and fewer computational demands|Lower computational and memory requirements|Low but inefficient on parallel hardware|Generally low but depends on graph size|
|Memory Usage|Higher memory usage due to full sequence attention|Lower memory usage due to their simple structure and fewer parameters|Lower memory usage due to localized operations|Low but can increase with sequence length|Generally low but depends on graph size|
|Parallel Processing|Highly parallelizable but requires synchronization due to attention mechanisms|Highly parallelizble due to independent computations|Highly parallelizable due to independent convolution operations|Difficult due to sequence dependencies|Difficult due to graph structure dependencies|

#### Table 7: Taxonomy and Methodologies of Traditional Deep Learning Architectures for Time Series Forecasting

7.1 MLP-Base

|Model Name|Main Methodology|Channel Correlatoin|Pub|
|:--------:|:---------------|:-----------------:|--:|
|Koopa|Koopa Block with Koopman Predictor(KP)|CD|2023|
|TSMixer|MLP Mixer layer architecture, Gated attention (GA) block, Online hierarchical patch reconciliation head|CI/CD|2023|
|FreTS|Frequency-domain MLPs|CD|2023|
|TSP|PrecMLP block with precurrent mechanism|CD|2024|
|FITS|Complex Frequency Linear Interpolation, Lower Pass Filter(LPF)|CI|2024|
|U-Mixer|Unet Encoder-decoder with MLPs, Stationarity Correlation|CD|2024|
|TTMs|Multi-Resolution Pre-training via TTM Backbone (TSMixer blocks), Exogenous mixer|CD|2024|
|TimeMixer|Past-Decomposable-Mixing (PDM) block, Future-Multipredictor-Mixing (FMM) block|CD|2024|
|CATS|Auxiliary Time Series (ATS) Constructor|CD|2024|
|HDMixer|Length-Extendable Patcher, Patch Entropy Loss, Hierarchical Dependency Explorer|CD|2024|
|SOFTS|Star Aggregate-Redistribute Module|CD|2024|
|SparseTSF|Cross-Period Sparse Forecasting|CI|2024|
|TEFN|Basic Probability Assignment (BPA) Module|CD|2024|
|PDMLP|Multi-Scale Patch Embedding & Feature Decomposition, Intra-, Inter-Variable MLP|CD|2024|
|AMD|Multi-Scale Decomposable Mixing(MDM) Block, Dual Dependency Interaction (DDI) Block, Adaptive Multi-predictor Synthesis (AMS) Block|CD|2024|

7.2 CNN-Base

|Model Name|Main Methodology|Channel Correlatoin|Pub|
|:--------:|:---------------|:-----------------:|--:|
|TimesNet|Transform 1D-variations into 2D-variation, Timesblock|CD|2023|
|PatchMixer|Patch Embedding, PatchMixer layer with Patch-mixing Design|CI|2023|
|ModernTCN|ModernTCN block with DWConv & ConvFFN|CD|2024|
|ConvTimeNet|Deformable Patch Embedding, Fully Convolutional Blocks|CD|2024|
|ACNet|Temporal Feature Extraction Module, Nonlinear Feature Adaptive Extraction Module|CD|2024|
|FTMixer|Frequency Channel Convolution, Windowing Frequency Convolution|CD|2024|

7.3 RNN-Base

|Model Name|Main Methodology|Channel Correlatoin|Pub|
|:--------:|:---------------|:-----------------:|--:|
|PA-RNN|Mixture Gaussian Prior, Prior Annealing Algorithm|CI|2023|
|WITRAN|Horizontal Vertical Gated Seletive Unit, Recurrent Acceleration Network|CI|2023|
|SutraNets|Sub-series autoregressive networks|CI|2023|
|CrossWaveNet|Deep cross-decomposition, Dual-channel network|CD|2024|
|DAN|RepGen & RepMerg with Polar Representation Learning, Distance-weighted Multi-loss Mechanism, Kruskal-Wallis Sampling|CI|2024|
|RWKV-TS|RWKV Blocks with Multi-head WKV Operator|CD|2024|
|CONTIME|Bi-directional Continuous GRU with Neural ODE|CI|2024|

7.4 GNN/GCN-Base

|Model Name|Main Methodology|Channel Correlatoin|Pub|
|:--------:|:---------------|:-----------------:|--:|
|MSGNet|SacleGraph block with Scale Identification, Multi-scale Adaptive Graph Convolution, Multi-head Attention and Scale Aggregation|CD|2024|
|TMP-Nets|Temporal MultiPersistence (TMP) Vectorization|CD|2024|
|HD-TTS|Temporal processing module with Temporal message passing, Spatial processing module with Spatial message passing|CD|2024|
|ForecastGrapher|Group Feature Convolution GNN (GFC-GNN)|CD|2024|

7.5 Hybird-Base

|Model Name|Main Methodology|Channel Correlatoin|Pub|
|:--------:|:---------------|:-----------------:|--:|
|WaveForM: CNN+GCN|Discrete Wavelet Transform (DWT) Module, Graph-Enhanced Prediction Module|CD|2023|
|BiTGraph: CNN+GCN|Multi-Scale Instance PartialTCN (MSIPT) Module, Biased GCN Module|CD|2024|
|TSLANet: TF+CNN|Adaptive Spectral Block, Interactive Convolutional Block|CI|2024|
|DERITS: CNN+MLP|Frequency Derivative Transformation (FDT), Order-adpative Fourier Convolution Network (OFCN)|CD|2024|

7.6 Model-agnostic

|Model Name|Main Methodology|Channel Correlatoin|Pub|
|:--------:|:---------------|:-----------------:|--:|
|RobustTSF|RobustTSF Algorithm|-|2024|
|PDLS|Loss Shaping Constraints, Empirical Dual Resilient and Constrained Learning|-|20244|
|Leddam|Learnable Decomposition Module, Dual Attention Module|CD|2024|
|InfoTime|Cross-Variable Decorrelation Aware Feature Modeling (CDAM), Temporal Aware Modeling (TAM)|CD|2024|
|CCM|Channel Clustering & Cluster Loss, Cluster-aware Feed Forwared|CD|2024|
|HCAN|Uncertainty-Aware Classifier (UAC), Hierarchical Consistency Loss (HCL), Hierarchy-Aware Attention (HAA)|-|2024|
|TDT Loss|Temporal Dependencies among Targets(TDT) Loss|-|2024|

#### Table 8: Taxonomy and Methodologies of Foundation Models for Time Series Forecasting

8.1 Sequential modeling with LLM Approach

|Model Name|Main Improvement & Methodology|Pub|
|:--------:|:-----------------------------|--:|
|GPT4TS|Demonstractie the effectiveness of LLM for time series modeling, Fine-tune the layer normalization and positional embedding parameters|2023|
|PromptCast|Enable text-level domain-specific knowledge for TSF, Cast TSF problem into question and answering format|2023|
|LLMTime|Zero-shot TSF with pre-trained LLMs, Covert time series input into a string of digits|2023|
|Time-LLM|Align time series modality into text modality, Convert time series input into a string of digits|2024|

8.2 Pre-training Approach

|Model Name|Main Improvement & Methodology|Pub|
|:--------:|:-----------------------------|--:|
|Lag-Llama|First pre-training based time series foundation model, Pre-train a decoder-only model with autoregressive loss|2024|
|TimesFM|Pre-trained with hundreds of billions time steps, Autoregressive decoding with arbitrary forecasting length|2024|
|CHRONOS|Learning the language of time series, Utilize tokenizer to capture the intrinsic language of time series|2024|
|UniTS|Explicit consideration of multivariate TSF, Provide variate IDs to directly consider multiple variables|2024|

#### Table 9: Taxonomy and Methodologies of Diffusion Models for Time Series Forecasting

9.1 Effective Conditional Embedding - Main Improvement

|Model Name|Main Methodology|Diffusion Type|Conditional Type|Pub|
|:--------:|:---------------|:------------:|:--------------:|--:|
|TimeGrad|Autoregressive DDPM using RNN & Dilated Convolution|DDPM|Explicit|2021|
|CSDI|2D Attention for Temporal & Feature Dependency, Self-supervised Training for Imputation|DDPM|Explicit|2021|
|SSSD|Combination of S4 model|DDPM|Explicit|2023|
|TimeDiff|Future Mixup, Autoregressive Initialization|DDPM|Explicit|2023|
|TMDM|Integration of Diffusion and Transformer-based Models|DDPM|Explicit|2024|

9.2 Time-series Feature Extraction - Main Improvement

|Model Name|Main Methodology|Diffusion Type|Conditional Type|Pub|
|:--------:|:---------------|:------------:|:--------------:|--:|
|Diffusion-TS|Decomposition techniques, Instance-aware Guidance Strategy|DDPM|Guidance|2024|
|Diffusion in Frequency|Diffusing in the Frequency Domain|SDE|Explicit|2024|
|MG-TSD|Multi-granularity Data Generator, Temporal Process Module, Guided Diffusion Process Module|DDPM|Explicit|2024|
|mr-Diff|Integration of Decomposition and Multiple Temporal Resolutions|DDPM|Explicit|2024|

9.3 SDE - Main Improvement

|Model Name|Main Methodology|Diffusion Type|Conditional Type|Pub|
|:--------:|:---------------|:------------:|:--------------:|--:|
|ScoreGrad|Continuous Energy-based Generative Model|SDE|Explicit|2021|
|D^3 M|Decomposable Denoising Diffusion Model based on Explicit Solutions|SDE|Explicit|2024|

9.4 Latent Diffusion Model - Main Improvement

|Model Name|Main Methodology|Diffusion Type|Conditional Type|Pub|
|:--------:|:---------------|:------------:|:--------------:|--:|
|LDT|Symmetric Time Series Compression, Latent Diffusion Transformer|DDPM|Guidance|2024|

9.5 Guidance - Main Improvement

|Model Name|Main Methodology|Diffusion Type|Conditional Type|Pub|
|:--------:|:---------------|:------------:|:--------------:|--:|
|TSDiff|Observation Self-guidance|DDPM|Guidance|2023|

#### Table 10: Taxonomy and Methodologies of Mamba Models for Time Series Forecasting

10.1 Embedding and Multi-Scale Learning - Main Improvement

|Model Name|Main Methodology|Channel Correlation|Base|Pub|
|:--------:|:---------------|:-----------------:|:--:|--:|
|TimeMachine|Integrated Quadruple Mambas|CD|Mamba|2024|

10.2 Channel Correlation Learning - Main Improvement

|Model Name|Main Methodology|Channel Correlation|Base|Pub|
|:--------:|:---------------|:-----------------:|:--:|--:|
|S-Mamba|Channel Mixing: Mamba VC Encoding Layer|CD|Mamba, MLP|2024|
|SiMBA|Channel Mixing: Einstein FFT (EinFFT), Sequence Modeling: Mamba|CD|Mamba|2024|
|MambaTS|Temporal Mamba Block (TMB)|CD|Mamba|2024|
|C-Mamba|Channel Mixup, C-Mamba Block (PatchMamba + Channel Attention)|CD|Mamba|2024|

10.3 Sequence Information and Dependency Learning - Main Improvement

|Model Name|Main Methodology|Channel Correlation|Base|Pub|
|:--------:|:---------------|:-----------------:|:--:|--:|
|Mambaformer|Mambaformer (Attention + Mamba) Layer|CI|Mamba, Transformer|2024|
|Bi-Mamba+|Series-Relation-Aware (SRA) Decider, Mamba+ Block, Bidirectional Mamba+ Encoder|CI/CD|Mamba|2024|
|DTMamba|Dual Twin Mamba Blocks|CI|Mamba|2024|

10.4 Theoretical Frameworks and Efficient Modeling - Main Improvement

|Model Name|Main Methodology|Channel Correlation|Base|Pub|
|:--------:|:---------------|:-----------------:|:--:|--:|
|Time-SSM|Dynamic Spectral Operator with Hippo-LegP|CD|Mamba|2024|
|Chimera|2-Dimensional State Space Model|CD|Mamba|2024|

Transformer Models: 2017년에 소개된 딥러닝 아키텍처로, 자연어 처리(NLP) 분야에서 혁신을 가져왔습니다. 특징으로는, 1. 어텐션 메커니즘: 입력 시퀀스 내의 토큰 간 관계를 포착하여, 모델이 중요한 정보에 집중할 수 있도록 합니다. 2. 병렬 처리: RNN과 달리, 시퀀스의 모든 부분을 동시에 처리할 수 있어 계산 효율성이 높습니다. 3. 장기 의존성 처리: 어텐션 메커니즘 덕분에, 긴 시퀀스 내의 장기 의존성을 효과적으로 처리할 수 있습니다.

Multi-Layer Perceptron (MLP): 간단한 구조의 인공 신경망으로, 여러 개의 은닉층을 가집니다. 장점으로는, 구현이 쉽고, 계산 효율성이 높습니다. 단점으로는, 순차적 데이터를 처리하는 데는 한계가 있으며, 장기 의존성을 포착하는 데 어려움이 있습니다.

Convolutional Neural Network (CNN): 주로 이미지 처리에 사용되지만, 시계열 데이터에도 적용됩니다. 장점으로는, 로컬 패턴을 효과적으로 추출할 수 있으며, 계산 효율성이 높습니다. 단점으로는, 장기 의존성을 포착하는 데는 한계가 있습니다.

Recurrent Neural Network (RNN): 순차 데이터를 처리하도록 설계되었으며, 시간적 종속성을 모델링할 수 있습니다. 장점으로는, 시계열 데이터의 시간적 패턴을 학습할 수 있습니다. 단점으로는, vanishing gradient 문제로 인해 장기 의존성을 학습하는 데 어려움이 있습니다.

Graph Neural Network (GNN): 그래프 구조의 데이터를 처리하며, 노드 간의 복잡한 관계를 모델링합니다. 장점으로는, 다변량 시계열 데이터의 변수 간 관계를 효과적으로 포착할 수 있습니다. 단점으로는, 그래프 구조의 복잡성에 따라 계산 비용이 증가할 수 있습니다.

Foundation Models: 대규모 데이터셋에서 사전 학습된 모델로, 다양한 작업에 적용할 수 있는 일반적인 능력을 가지고 있습니다. 특징으로는, 1. 대규모 사전 학습: 방대한 양의 데이터에서 학습하여, 일반적인 지식을 습득합니다. 2. 전이 학습: 사전 학습된 모델을 특정 작업에 맞게 미세 조정하여, 데이터가 부족한 상황에서도 높은 성능을 발휘할 수 있습니다. 3. 제로샷 및 퓨샷 학습: 일부 학습에서는 사전 학습된 모델을 전혀 미세 조정하지 않거나, 소량의 데이터만으로 학습할 수 있습니다.

Diffusion Models: 생성 모델의 한 종류로, 데이터 생성 과정에서 노이즈를 점진적으로 추가하고 제거하는 과정을 통해 데이터를 생성합니다. 특징으로는, 1. 노이즈 주입 및 제거: 데이터에 노이즈를 점진적으로 추가하는 전방향 과정과, 노이즈를 제거하여 원본 데이터를 복구하는 역방향 과정으로 구성됩니다. 2. 확률적 생성: 확률적 과정을 통해 데이터를 생성하므로, 다양한 샘플을 생성할 수 있습니다. 3. 고품질 생성: 이미지, 오디오, 비디오 등 다양한 데이터 타입에서 고품질의 데이터를 생성할 수 있습니다.

Mamba Models: 최근에 등장한 새로운 딥러닝 아키텍처로, 상태 공간 모델(State Space Model, SSM)을 기반으로 합니다. 특징으로는, 1. 선택적 SSM (Selective SSM): 입력 데이터에 따라 SSM의 매개변수가 동적으로 변경되도록 하여, 특정 부분의 시퀀스를 선택적으로 기억하거나 무시할 수 있습니다. 2. 선형 계산 복잡성: Transformer의 이차적 계산 복잡성과 달리, Mamba는 선형 계산 복잡성을 가지므로 긴 시퀀스를 처리하는 데 효율적입니다. 3. 병렬 처리: SSM의 특성상, 시스템의 매개변수 행렬이 시간에 따라 변하지 않으므로, 전역적으로 적용 가능한 커널을 미리 계산할 수 있어 병렬 처리가 가능합니다. 4. 단순화된 아키텍처: 어텐션 메커니즘이나 별도의 MLP 블록이 없으므로, 계산 복잡성이 줄어들고, 효율적인 학습과 빠른 추론이 가능합니다.

#### 5 TSF Latest Open Challenges & Handling Methods

* 5.1 Channel Dependency Comprehension > Fig. 15: Comparison of CI and CD Strategies in Channel  Correlations
  * Spread of Channel Independent Strategy
  * Importance of Learning Channel Correlations
  * What Makes CI Look Better?
  * Recent Approaches > Fig. 16: Recent Approaches to Channel Strategies
* 5.2 Alleviation of Distribution Shift: DAIN, RevIN, NST, Dish-TS, SAN > Table 11: Normalizaion-Denormalization-based Approaches to Alleviate Distribution Shifts in Time Series Forecasting
* 5.3 Enhancing Causality
  * Why Casusal Analysis is Essential for Accurate Time Series Forecasting
  * Research on TSF with Causality: Kuroshio Volume Transport (KVT), GCN with ConvLSTM, Granger causality test with Bi-LSTM, Causal-GNN using SIRD Attention-Based Dynamic GNN, Caformer.
* 5.4 Time Series Feature Extraction
  * Understanding the characteristics of data
  * Explainability of data
  * Enhancing Model Performance
  * 5.4.1 Decomposition
    * Moving Average Kernel: Autoformer, CrossWaveNet, FEDformer, LTSF-Linear, PDMLP, Leddam, Diffusion-TS.
    * Downsampling: SparseTSF, SutraNets.
  * 5.4.2 Multi-scale: MTST, PDMLP, FTMixer, TimeMixer, AMD, HD-TTS, Scaleformer, Pathformer, MG-TSD, mr-Diff
  * 5.4.3. Domain transformation
    * Periodicity Extraction: Autoformer, TimesNet, MSGNet
    * Training in the Frequency Domain: FreTS, FEDformer, Fredformer, FITS, DERITS, SiMBA, WaveForM, FTMixer
  * 5.4.4 Aiddtional approach: CATS, SOFTS
 
Channel Correlation (채널 상관관계): 다변량 시계열 데이터에서 여러 변수(채널) 간의 관계를 나타냅니다. 다련량 시계열 데이터는 여러 개의 변수가 동시에 관측되는 데이터를 의미하며, 이러한 병수들은 서로 독립적일 수 있고, 서로 영향을 미치며 상관관계를 가질 수도 있습니다. 예를 들어, 기상 데이터에서 온도와 전력 소비 간에는 강한 상관관계가 있을 수 있습니다. 

Channel Independence (CI, 채널 독립성): 다변량 시계열 데이터에서 각 변수(채널)기 서로 독립적이라고 가정하는 전략입니다. 즉, 변수들 간의 상관관계를 무시하고 각 채널을 개별적으로 모델링합니다. 특징으로는, 1. 단순성: CI 전략은 모델을 단순화하여 계산 효율성을 높입니다. 각 채널을 독립적으로 처리하기 때문에, 변수 간의 복잡한 상관관계를 모델링할 필요가 없습니다. 2. 강건성: 변수 간의 상관관계가 불규칙하거나 노이즈가 많은 경우, CI 전략은 이러한 문제를 피할 수 있어 더 강건한 모델을 제공할 수 있습니다. 3. 적용성: CI 전략은 번수 간의 상관관계가 크지 않거나, 상관관계가 모델의 예측 성능에 큰 영향을 미치지 않는 경우에 적합합니다. 한계로는 변수 간의 중요한 상관관계를 무시할 수 있으며, 이는 예측 성능을 저하시킬 수 있습니다.

Channel Dependency (CD, 채널 종속성): 다변량 시계열 데이터에서 변수들 간의 상관관계를 명시적으로 모델링하는 전략입니다. 즉, 변수들 간의 상호작용과 관계를 고려하여 모델을 설계합니다. 특징으로는, 1. 복잡성: CD 전략은 변수 간의 상관관계를 모델링하기 때문에, 모델의 복잡성이 증가합니다. 이는 더 높은 계산 비용과 더 복잡한 모델 구조를 필요로 합니다. 2. 성능 향상: 변수 간의 인과 관계가 명확한 경우, CD 전략은 더 정확한 예측을 제공할 수 있습니다. 3. 적용성: 변수 간의 상관관계가 크고, 이러한 관계가 모델의 예측 성능에 중요한 영향을 미치는 경우에 적합합니다. 한계로는 모델의 복잡성이 증가하여 과적합의 위험이 있으며, 데이터가 충분하지 않을 경우 성능이 저하될 수 있습니다.

요약:
* CR: 변수 간의 관계를 나타내며, 예측 모델의 성능에 중요한 영향을 미칩니다.
* CI: 변수 간의 상관관계를 무시하고 각 채널을 독립적으로 모델링하는 전략입니다.
* CD: 변수 간의 상관관계를 명시적으로 모델링하여 예측 성능을 향상시키는 전략입니다.

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
