# Artificial Intelligence (AI) | [IBM](https://www.ibm.com/cloud/learn/what-is-artificial-intelligence)

Artificial intelligence leverages computers and machines to mimic the problem-solving and decision-making capabilities of the human mind.

## Weak AI | Narrow AI | Artificial Narrow Intelligence (ANI)
Weak AI is AI trained and focused to perform specific tasks. Weak AI drives most of the AI that surrounds us today. 'Narrow' might be a more accurate descriptor for this type of AI as it is anyhing but weak; it enables some very robust applications, such as Apple's Siri, Amazon's Alexa, IBM Watsom, and autonomous vehicles.

## Strong AI | Artificial General Intelligence (AGI) | Artificial Super Intelligence (ASI) | General AI

Strong AI is made up of Artificial General Intelligence (AGI) and Artificial Super Intelligence (ASI). Artificial general intelligence (AGI), or general AI, is a theoretical form of AI where a machine would have an intelligence equaled to humans; it would have a self-aware consciousness that has the ability to solve problems, learn, and plan for the future. Artificial Super Intelligence (ASI)-also known as superintelligence-would surpass the intelligence and ability of the human brain. While storng AI is still entirely theoretical with no practical examples in use today, that doesn't mean AI researchers aren't also exploring its development. In the meantime, the best examples of ASI might be from science fiction, such as HAL, the superhuman, rogue computer assistant in 2001: A Space Odyssey.

## Machine Learning (ML)

IBM research suggest ML to evolve around the following segments in the future:
- Classic ML: Deals with problems with one dataset at a time, one task and one heavy training.
- Few-shot ML: Deals with heavy offline training, then easy learning on similar tasks.
- Developing ML: Continuous life-long learning on various tasks.

## Deep Learning (DL)

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

### Offline Learning | Batch Learning | [Blog (KR)](https://irron2004.tistory.com/2)

- Characteristic
  - Use all data to learn model
  - Need to use previous data to learn model with new data
- Pros
- Cons
  - More resources required
  
### Lifelong Learning | Continual Learning | [Blog (KR)](https://realblack0.github.io/2020/03/22/lifelong-learning.html)

Methods:
- Regularization: NN의 weight를 예전 task의 성능에 기여한 중요도에 따라 weight update를 제한 - 중요한 weight일 수록 semantic drift가 발생하지 않도록 하여 multi task가 가능
   - Elastic Weight Consolidation (EWC) by Google Deepmind
     - With no penalty, 

- Structure
  - Progressive Network

- Memory
  - Deep Generative Replay

- Fusion
  - Dynamically Expandable Network 

### Catastrophic Forgetting

- Characteristic
  - Performance reduced, when learning different task
  - Loss of information, even if there is an association between the old training dataset and the new training dataset

### Semantic Drift

- Characteristic
  - Node or wieght can be changed, when pre-trained weight has been excessively adjusted in the process of new learning

---

### [Mila](https://mila.quebec/en/mila/)

Founded in 1993 by Professor Yoshua Bengio of the Universite de Montreal, Mila is a research institue in artificial intelligence that rallies nearly 900 researchers specializing in the field of machine learning.

### [DeepMind](https://deepmind.com/)

### [OpenAI](https://openai.com/)

### [MIT CSAIL](https://www.csail.mit.edu/)

---

### [GitHub Copilot](https://copilot.github.com/)

### [OpenAI Codex](https://openai.com/blog/openai-codex/)

Codex is the model that powers GitHub Copilot.

---

### Articles related to AI

- [인공일반지능(AGI), 시도해도 될까](https://www.technologyreview.kr/artificial-general-intelligence-robots-ai-agi-deepmind-google-openai/), 2022-02-13-Sun.
- [Artificial general intelligence: Are we close, and does it even make sense to try?]( https://www.technologyreview.com/2020/10/15/1010461/artificial-general-intelligence-robots-ai-agi-deepmind-google-openai/), 2022-02-13-Sun.
- [Meta's new learning algorithm can teach AI to multi-task](https://www.technologyreview.com/2022/01/20/1043885/meta-ai-facebook-learning-algorithm-nlp-vision-speech-agi/), 2022-02-14-Mon.
- [Machine that learn language more like kids do](https://www.csail.mit.edu/news/machines-learn-language-more-kids-do), 2022-02-17-Thu.
- [Competitive programming with AlphaCode](https://deepmind.com/blog/article/Competitive-programming-with-AlphaCode), 2022-02-18-Fri.
- [Using reinforcement learning to identify high-risk states and treatments in healthcare](https://www.microsoft.com/en-us/research/blog/using-reinforcement-learning-to-identify-high-risk-states-and-treatments-in-healthcare/), 2022-02-20-Sun.
- [AlphaFold: a solution to a 50-year-old grand challenge in biology](https://deepmind.com/blog/article/alphafold-a-solution-to-a-50-year-old-grand-challenge-in-biology), 2022-02-21-Mon.
- [Can machine-learning models overcome biased datasets?](https://news.mit.edu/2022/machine-learning-biased-data-0221), 2022-02-22-Tue.
- [Can Computer Algorithms Learn about the Ethics of Warfare?](https://www.technologyreview.kr/%ec%bb%b4%ed%93%a8%ed%84%b0-%ec%95%8c%ea%b3%a0%eb%a6%ac%eb%93%ac%ec%9d%80-%ec%9c%a4%eb%a6%ac%ec%a0%81-%ec%a0%84%ed%88%ac%eb%b0%a9%eb%b2%95%ec%9d%84-%eb%b0%b0%ec%9a%b8-%ec%88%98-%ec%9e%88%ec%9d%84/), 2022-03-07-Mon.
- [Goole's Artificial Intelligence App LaMDA Believes It's Human With Real Thoughts & Feelings](https://globalgrind.com/5342206/google-lambda-sentience/amp/), 2022-06-23-Thu.
- [In-home wireless device tracks disease progression in Parkinson's patients](https://news.mit.edu/2022/home-wireless-parkinsons-progression-0921), 2022-09-26-Mon.
- [LOLNeRF: Learn from One Look](https://ai.googleblog.com/2022/09/lolnerf-learn-from-one-look.html), 2022-09-26-Mon.
- [AL that can learn the patterns of human language](https://feedly.com/i/entry/NwHO2cNXnJxomKwaSvhGDBXV4Lc7B4INaC4YnMl3/fs=_182ef0744b0:4878f3:559ea8bd), 2022-09-26-Mon.

### Articles unrelated to AI

- [미래의 안보 기술을 찾아라…美 정보기관 NSA에 떨어진 특명](https://www.technologyreview.kr/%eb%af%b8%eb%9e%98%eb%a5%bc-%eb%a7%8c%eb%93%a4%ea%b3%a0-%ec%9e%88%eb%8a%94-%eb%af%b8%ea%b5%ad-nsa%ec%9d%98-%ec%8a%a4%ed%8c%8c%ec%9d%b4%eb%93%a4/), 2022-02-15-Tue.
- [Robust Routing Using Electrical Flows](https://ai.googleblog.com/2022/02/robust-routing-using-electrical-flows.html), 2022-02-19-Sat.

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

---

### Reference
- What is artificial intelligence IBM, https://www.ibm.com/cloud/learn/what-is-artificial-intelligence, 2022-02-12-Sat.
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
- Lifelong Learning Blog KR, https://realblack0.github.io/2020/03/22/lifelong-learning.html, 2022-11-09-Wed.
