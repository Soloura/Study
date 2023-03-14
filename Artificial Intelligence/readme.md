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

## Deep Learning | [IBM](https://www.ibm.com/topics/deep-learning)

Deep learning is a subset of machine learning, which is essentially a neural network with three or more layers. These neural networks attempt to simulate the behavior of the human brain-albeit far matching its ability-allowing it to learn from large amounts of data. While a neural network with a single layer can still make approximate predictions, additional hidden layers can help to optimize and refine for accuracy.

Deep learning is actually comprised of neural networks. "Deep" in deep learning refers to a neural network comprised of more than three layers-which would be inclusive of the inputs and the output-can be considered a deep learning algorithm.

## Neural Modeling Fields | [WiKi](https://en.wikipedia.org/wiki/Neural_modeling_fields)

Neural Modeling Field (NMF) is a mathematical framework for machine learning which combines ideas from neural networks, fuzzy logic, and model based recognition. It has also been referred to as modeling fields, modeling fields theory (MFT), Maximum likelihood artificial neural networks (MLANS). This framework has been developed by Leonid Perlovsky at the AFRL. NMF is interpreted as a mathematical description of mind's mechanisms, including concepts, emotions, instincts, imagination, thinking, and understanding. NMF is a multi-level, hetero-hierarchical system. At each level in NMF there are concept-models encapsulating the knowledge; they generate so-called top-down signals, interacting with input, bottom-up signals. THese interactions are governed by dynamic equations, which drive concept-model learning, adaptation, and formation of new concept-models for better correspondence to the input, bottom-up signals.

### Casual Learning | [Distributed Training](https://distributedtraining.com/casual-learning/)

Casual Learning is a more relaxted approach to finding out how to do new things.

Learn at your own pace, in your own time and just what you want to know about, not what the rigid course director says you must have completed before you get to the bit you;re actually interested in.

### Online Learning | [Blog (KR)](https://daeson.tistory.com/225)

- Characteristic
  - Learn model with sequencial data or mini-batch data
  - Learn model with new data
- Pros
  - Fast
  - Less resources
- Cons
  - According to learning rate, recent or past pattern can be forgotten

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

### Catastrophic Forgetting

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

##

### [GitHub Copilot](https://copilot.github.com/)

### [OpenAI Codex](https://openai.com/blog/openai-codex/)

Codex is the model that powers GitHub Copilot.

---

### MLOps | [Databricks](https://www.databricks.com/glossary/mlops) | [Google Cloud](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

MLOps stands for Machine Learning Operations. MLOps is a core function of Machine Learning engineering, focused on streamlining the process of taking machine learning models to production, and then maintaining and monitoring them. MLOps is a collaborative function, often comprising data scientists, devops engineers, and IT.

Continuous delivery and automation pipelines in machine learning

This document discusses techniques for implementing and automating continuous integration (CI), continuous delivery (CD), and continuous training (CT) for machine learning (ML) systems.

Data science and ML are becoming core capabilities for solving complex real-world problems, transforming industires, and delivering value in all domains. Currently, the ingredients for applying effective ML are available to you:

- Large datasets
- Inexpensive on-demand compute resources
- Specialized accelerators for ML on various cloud platforms
- Rapid advances in different ML research fields (such as computer vision, natural language understanding, and recommendations AI systems).

Therefore, many business are investing in their data science teams and ML capabilities to develop predictive models that can deliver business value to their users.

This document is for data scientists and ML engineers who want to apply DevOps principles to ML systems (MLOps). MLOps is an ML engineering culture and practice that aims at unifying ML system development (Dev) and ML system operation (Ops). Practicing MLOps means that you advocate for automation and monitoring at all steps of ML system construction, including integrations, testing, releasing, deployment and infrastructure management.

Data scientists can implement and train an ML model with predictive performance on an offline holdout dataset, given relevant training data for their use case. However, the real challenge isn't building an ML model, the challenge is building an integrated ML system and to continuously operate it in production. With the long history of production ML services at Google, we've learned that there can be many pitfalls in operating ML-based systems in production. Some of these pitfalls are summarized in Machine Learning: The high-interest credit card of technical debt.

Only a small fraction of a real-world ML system is composed of the ML code. The required surrounding elements are vast and complex.

The rest of the system is composed of configuration, automation, data collection, data verification, testing and debugging, resource management, model analysis, process and metadata management, serving infrastructure, and monitoring.

To develop and operate complex systems like these, you can apply DevOps principles to ML systems (MLOps). This document covers concepts to consider when setting up an MLOps environment for your data science practices, such as CI, CD, and CT in ML.

### Continous Machine Learning | [Blog](https://levity.ai/blog/what-is-continuous-machine-learning)

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
- MLOps Databricks, https://www.databricks.com/glossary/mlops, 2022-10-04-Tue.
- Casual Learning Distributed Training, https://distributedtraining.com/casual-learning/, 2022-10-07-Fri.
- Online Learning Blog KR, https://daeson.tistory.com/225, 2022-10-07-Fri.
- MLOps Google Cloud, https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning, 2022-10-11-Tue.
- Batch Learning Offline Learning Online Learning Blog KR, https://irron2004.tistory.com/2, 2022-11-09-Wed.
- Federated Learning WiKi, https://en.wikipedia.org/wiki/Federated_learning, 2022-12-17-Sat.
- Continous Machine Learning Blog, https://levity.ai/blog/what-is-continuous-machine-learning, 2023-03-14-Tue.
- Offline Learning Blog, https://www.aiforanyone.org/glossary/offline-learning, 2023-03-14-Tue.
