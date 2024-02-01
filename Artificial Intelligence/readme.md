# Artificial Intelligence | [IBM](https://www.ibm.com/cloud/learn/what-is-artificial-intelligence)

Artificial intelligence leverages computers and machines to mimic the problem-solving and decision-making capabilities of the human mind.

### Weak AI | Narrow AI | Artificial Narrow Intelligence (ANI)

Weak AI-also called Narrow AI or Artificial Narrow Intelligence (ANI)-is AI trained and focused to perform specific tasks. Weak AI drives most of the AI that surrounds us today. 'Narrow' might be a more accurate descriptor for this type of AI as it is anything but weak; it enables some very robust applications, such as Apple's Siri, Amazon's Alexa, IBM Watsom, and autonomous vehicles.

### Strong AI | Artificial General Intelligence (AGI) | Artificial Super Intelligence (ASI) | General AI

Strong AI is made up of Artificial General Intelligence (AGI) and Artificial Super Intelligence (ASI). Artificial general intelligence (AGI), or general AI, is a theoretical form of AI where a machine would have an intelligence equaled to humans; it would have a self-aware consciousness that has the ability to solve problems, learn, and plan for the future. Artificial Super Intelligence (ASI)-also known as superintelligence-would surpass the intelligence and ability of the human brain. While storng AI is still entirely theoretical with no practical examples in use today, that doesn't mean AI researchers aren't also exploring its development. In the meantime, the best examples of ASI might be from science fiction, such as HAL, the superhuman, rogue computer assistant in 2001: A Space Odyssey.

## Machine Learning | [IBM](https://www.ibm.com/topics/machine-learning)

Machine learning is a branch of artificial intelligence (AI) and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy.

IBM research suggest ML to evolve around the following segments in the future:
- Classic ML: Deals with problems with one dataset at a time, one task and one heavy training.
- Few-shot ML: Deals with heavy offline training, then easy learning on similar tasks.
- Developing ML: Continuous life-long learning on various tasks.

[Boosting](https://m.blog.naver.com/laonple/220834569716)

[Bagging](https://m.blog.naver.com/laonple/220838501228)

[SVM](https://m.blog.naver.com/laonple/220845107089)

## Deep Learning | [IBM](https://www.ibm.com/topics/deep-learning)

Deep learning is a subset of machine learning, which is essentially a neural network with three or more layers. These neural networks attempt to simulate the behavior of the human brain-albeit far matching its ability-allowing it to learn from large amounts of data. While a neural network with a single layer can still make approximate predictions, additional hidden layers can help to optimize and refine for accuracy.

Deep learning is actually comprised of neural networks. "Deep" in deep learning refers to a neural network comprised of more than three layers-which would be inclusive of the inputs and the output-can be considered a deep learning algorithm.

## Neural Modeling Fields | [WiKi](https://en.wikipedia.org/wiki/Neural_modeling_fields)

Neural Modeling Field (NMF) is a mathematical framework for machine learning which combines ideas from neural networks, fuzzy logic, and model based recognition. It has also been referred to as modeling fields, modeling fields theory (MFT), Maximum likelihood artificial neural networks (MLANS). This framework has been developed by Leonid Perlovsky at the AFRL. NMF is interpreted as a mathematical description of mind's mechanisms, including concepts, emotions, instincts, imagination, thinking, and understanding. NMF is a multi-level, hetero-hierarchical system. At each level in NMF there are concept-models encapsulating the knowledge; they generate so-called top-down signals, interacting with input, bottom-up signals. THese interactions are governed by dynamic equations, which drive concept-model learning, adaptation, and formation of new concept-models for better correspondence to the input, bottom-up signals.

### Casual Learning | [Distributed Training](https://distributedtraining.com/casual-learning/)

Casual Learning is a more relaxted approach to finding out how to do new things.

Learn at your own pace, in your own time and just what you want to know about, not what the rigid course director says you must have completed before you get to the bit you're actually interested in.

### Causal Learning

[Introduction to Causal Inference (Brady Neal)](https://www.bradyneal.com/causal-inference-course)

[Blog (KR)](https://brunch.co.kr/@advisor/23)

BabyAI: A Platform to Study the Sample Efficiency of Grounded Language Learning | ICLR 2019 | [arXiv](https://arxiv.org/abs/1810.08272) | [GitHub](https://github.com/mila-iqia/babyai)

Reccurent Indepedent Mechanisms | ICLR 2021 | [arXiv](https://arxiv.org/abs/1909.10893) | [GitHub](https://github.com/anirudh9119/RIMs)

### Online Learning | [Wiki](https://en.wikipedia.org/wiki/Online_machine_learning)

Online machine learning is a method of machine learning in which data becomes available in a sequential order and is used to update the best predictor for future data at each step, as opposed to batch learning techniques which generate the best predictor by learning on the entire training data set at once. Online learning is a common technique used in areas of machine learning where it is computationally infeasible to train over the entire dataset, requiring the need of out-of-core algorithms. It is also used in situations where it is necessary for the algorithm to dynamically adapt to new patterns in the data, or when the data itself is generated as a function of time, e.g., stock price prediction. Online learning algorithms may be prone to catastrophic interference, a problem that can be addressed by incremental learning approaches.

- Characteristic
  - Learn model with sequencial data or mini-batch data
  - Learn model with new data
- Pros
  - Fast
  - Less resources
- Cons
  - According to learning rate, recent or past pattern can be forgotten

[Blog (KR)](https://daeson.tistory.com/225)

### Incremental Learning | [Wiki](https://en.wikipedia.org/wiki/Incremental_learning)

Incremental learning is a method of machine learning in which input data is continuously used to extend the existing model's knowledge i.e. to further train the model. It represents a dynamic technique of supervised learning and unsupervised learning that can be applied when training data becomes available gradually over time or its size is out of system memory limits. Algorithms that can facilitate incremental learning are known as incremental machine learning algorithms.

Many traditional machine learning algorithms inherently support incremental learning. Other algorithms can be adapted to facilitate incremental learning. Examples of incremental algorithms include decision trees, decision rules, artificial neural networks or the incremental SVM.

The aim of incremental learning is for the learning model to adapt to new data without forgetting its existing knowledge. Some incremental learners have built-in some parameter or assumption that controls the relevancy of old data, while others, called stable incremental machine learning algorithms, learn representations of the training data that are not even partially forgotten over time. 

Incremental algorithms are frequently applied to data streams or big data, addressing issues in data availability and resource scarcity respectively. Stock trend prediction and user profiling are some examples of data streams where new data becomes continuously available. Applying incremental learning to big data aims to produce faster classification or forecasting times.

### Offline Learning | Batch Learning | [AI for Anyone](https://www.aiforanyone.org/glossary/offline-learning) | [Blog (KR)](https://irron2004.tistory.com/2)

Offline learning is a type of AI where the system is not constantly being trained with new data. Instead, it is trained with a set of data and then left to learn on its own.

- Characteristic
  - Use all data to learn model
  - Need to use previous data to learn model with new data
- Pros/Benefits
  - Reduce the amount of data that is needed to train a model
    - Learn from a smaller dataset and then transfer that knowledge to a larger dataset
    - Reduce the amount of time and resources
  - Improve the generalization of a model
    - Learn from a variety of data sources
    - Learn to generalize better to new data
  - Improve the interpretability of a model
    - Learn to better understand the data and the relationships between the data
- Cons
  - More resources required

### Lifelong Learning

`This content has moved to the 'Lifelong Learning' page.`

### Catastrophic Forgetting/Interference | [Wiki](https://en.wikipedia.org/wiki/Catastrophic_interference)

Catastrophic interference is the tendency of an artificial neural network to abruptly and drastically forget previously learned information upon learning new information. Neural networks are an important part of the network approach and connectionist approach to cognitive science. With these networks, human capabilities such as memory and learning can be modeled using computer simulations.

- Characteristic
  - Performance reduced, when learning different task
  - Loss of information, even if there is an association between the old training dataset and the new training dataset

### Semantic Drift

- Characteristic
  - Node or wieght can be changed, when pre-trained weight has been excessively adjusted in the process of new learning
  
### Federated Learning | [WiKi](https://en.wikipedia.org/wiki/Federated_learning)

Federated Learning (also known as collaborative learning) is a machine learning technique that trains an algorithm across multiple decentralized edge devices or servers holding local data samples, without exchaning them. This approach stands in contrast to traditional centralized machine learning techniques where all the local datasets are uploaded to one server, as well as to more classical decentralized approaches which often assume that local data samples are identically distributed.

Federated learning enables multiple actors to build a common, robust machine learning model without sharing data, thus allowing to address critical issues such as data privacy, data security, data access rights and access to heterogeneous data. Its applicaitons are spread over a number of industires including defences, telecommunications, IoT, and pharmaceutices. A major open question at the moment is how inferior models learned through federated data are relative to ones where the data are pooled. Another open question concerns the trustworthiness of the edge devices and the impact of malicious actors on the learned model.

---

## Organazation

### [Mila](https://mila.quebec/en/mila/)

Founded in 1993 by Professor Yoshua Bengio of the Universite de Montreal, Mila is a research institue in artificial intelligence that rallies nearly 900 researchers specializing in the field of machine learning.

### [DeepMind](https://deepmind.com/)

### [OpenAI](https://openai.com/)

### [MIT CSAIL](https://www.csail.mit.edu/)

---

## Programming Assist

### [GitHub Copilot](https://copilot.github.com/)

### [OpenAI Codex](https://openai.com/blog/openai-codex/)

Codex is the model that powers GitHub Copilot.

---

### Hysteresis | [Wiki](https://en.wikipedia.org/wiki/Hysteresis) | [Blog (KR)](https://m.blog.naver.com/yhy886700/221338210668)

Hysteresis is the dependence of the state of a system on its history.

이력 현상은 파괴된 뒤 결코 다시 회복할 수 없는 현상, 강한 비선형 현상이란 선형화가 단지 관찰된 형상만으로 분류할 수 없다는 것, 어떤 물리량이 그때의 물리 조건만으로는 일의적으로 결정되지 않고 그 이전에 그 물질이 경과해온 상태의 변화 과정에 의존하는 현상이다.

---

## :iphone: Services

### [ChatSonic](https://writesonic.com/chat?ref=soanle13&gclid=CjwKCAjwl6OiBhA2EiwAuUwWZWnBY_9cqxPVdoeCQLQG22w4FKA4-p2zGt8WUAnQl0RSU7Ho2diVZBoC_HoQAvD_BwE)

### [DeepL](https://www.deepl.com/translator): Translator

### [Craiyon](https://www.craiyon.com/): Image Generator

### [invideo](https://invideo.io/): Video Generator

### [upscale.media](https://www.upscale.media/): Upscale and Enhance Image

### [ContentIdeas](https://contentideas.io/): Images

---

### [Prediction]

Prediction refers to the output of an algorithm after it has been trained on a historical dataset and applied to new data when forecasting the likelihood of a particular outcome, such as whether or not a customer will churn in 30 days. The algorithm will generate probable values for an unknown variable for each record in the new data, allowing the model builder to identify what that value will most likely be.

The word prediction can be misleading. In some cases, it really does mean that you are predicting a future outcome, such as when you're using machine learning to determine the next best action in a marketing campaign. Other times, though, the prediction has to do with, for example, whether or not a transaction that already occurred was fraudulent. In that case, the transaction already happened, but you're making an educated guess about whether or not it was legitimate, allowing you to take the appropriate action.

---

### Reference
- Artificial Intelligence IBM, https://www.ibm.com/cloud/learn/what-is-artificial-intelligence, 2022-02-12-Sat.
- Machine Learning IBM, https://www.ibm.com/topics/machine-learning, 2023-03-14-Tue.
- Deep Learning IBM, https://www.ibm.com/topics/deep-learning, 2023-03-14-Tue.
- About Mila, https://mila.quebec/en/mila/, 2022-02-12-Sat.
- DeepMind, https://deepmind.com/, 2022-02-13-Sun.
- OpenAI, https://openai.com/, 2022-02-13-Sun.
- GitHub Copilot, https://copilot.github.com/, 2022-02-13-Sun.
- OpenAI Codex, https://openai.com/blog/openai-codex/, 2022-02-13-Sun.
- What is Few-Shot Learning? Methods & Applications in 2022, https://research.aimultiple.com/few-shot-learning/, 2022-02-28-Mon.
- Neural Modeling Fields Wiki, https://en.wikipedia.org/wiki/Neural_modeling_fields, 2022-09-19-Mon.
- Casual Learning Distributed Training, https://distributedtraining.com/casual-learning/, 2022-10-07-Fri.
- Online Learning Wiki, https://en.wikipedia.org/wiki/Online_machine_learning, 2023-03-14-Tue.
- Online Learning Blog KR, https://daeson.tistory.com/225, 2022-10-07-Fri.
- Incremental Learning Wiki, https://en.wikipedia.org/wiki/Incremental_learning, 2023-03-14-Tue.
- Batch Learning Offline Learning Online Learning Blog KR, https://irron2004.tistory.com/2, 2022-11-09-Wed.
- Federated Learning WiKi, https://en.wikipedia.org/wiki/Federated_learning, 2022-12-17-Sat.
- Offline Learning Blog, https://www.aiforanyone.org/glossary/offline-learning, 2023-03-14-Tue.
- Catastrophic Interference Wiki, https://en.wikipedia.org/wiki/Catastrophic_interference, 2023-03-14-Tue.
- Casual Learning Blog KR, https://brunch.co.kr/@advisor/23, 2023-04-10-Mon.
- Introduction to Casual Inference Brady Neal, https://www.bradyneal.com/causal-inference-course, 2023-04-18-Tue.
- Hysteresis Wiki, https://en.wikipedia.org/wiki/Hysteresis, 2023-04-19-Wed.
- Hysteresis Blog KR, https://m.blog.naver.com/yhy886700/221338210668, 2023-04-19-Wed.
- ChatSonic, https://writesonic.com/chat?ref=soanle13&gclid=CjwKCAjwl6OiBhA2EiwAuUwWZWnBY_9cqxPVdoeCQLQG22w4FKA4-p2zGt8WUAnQl0RSU7Ho2diVZBoC_HoQAvD_BwE, 2023-04-27-Thu.
- Translator DeepL, https://www.deepl.com/translator, 2023-04-27-Thu.
- Image Generator Craiyon, https://www.craiyon.com/, 2023-04-27-Thu.
- Video Generator invideo, https://invideo.io/, 2023-04-27-Thu.
- upscale media, https://www.upscale.media/, 2023-04-27-Thu.
- ContentIdeas, https://contentideas.io/, 2023-04-27-Thu.
- Boosting Blog KR, https://m.blog.naver.com/laonple/220834569716, 2023-11-06-Mon.
- Bagging Blog KR, https://m.blog.naver.com/laonple/220838501228, 2023-11-06-Mon.
- SVM Blog KR, https://m.blog.naver.com/laonple/220845107089, 2023-11-06-Mon.
- Prediction, https://www.datarobot.com/wiki/prediction/, 2024-02-01-Thu.
