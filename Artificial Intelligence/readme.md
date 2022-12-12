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
1. Regularization: NN의 weight를 예전 task의 성능에 기여한 중요도에 따라 weight update를 제한 - 중요한 weight일 수록 semantic drift가 발생하지 않도록 하여 multi task가 가능
   - Elastic Weight Consolidation (EWC) by Google Deepmind
     - Task에 대한 conditional probabilty를 사용
       - logP(sigma|D) = logp(D|sigma) + logp(sigma) - logp(D) = logp(Db|sigma) + logp(sigma|Da) - logp(Db)
     - Fi(Fisher Information Matrix)를 활용하여 weight parameter에 제한을 가하는 loss function을 사용
       - L(sigma) = Lb(sigma) + sum(lambda/2)Fi(sigmai-sigmaAi)**
     - Comparison 
       - With no penalty, weight는 old task의 최적화 범위를 벗어나 new task에만 최적화 -> semantic drift, catastrophic forgetting
       - With L2 regularization, new task의 성능을 포기하는 만큼 old task의 성능을 보존 -> multi task 모두 만족시키지 못하는 성능에 수렴
       - With EWC, old task 성능 유지하며 old task의 성능을 최대화 weight를 찾음 -> multi task 오차가 적은 교집합 부분으로 weight 갱신

2. Structure: NN의 구조를 동적으로 변경 - node/layer 추가
   - Progressive Network by Google Deepmind
     - Method
       - Transfer learning model은 pre-trained weight를 초기화 단계에서 통합
       - Progressive networks는 new task 학습 때 pre-trained를 남겨둠
       - New task 학습할 때 NN에 sub network을 추가하여 구조를 변경
       - Sub network은 new task를 학습하는데만 사용
       - Pre-trained weight로부터 유용한 feature 추출하여 sub network 학습에 활용
     - Training
       - Task 1 학습할 때, 기본 NN 사용
       - Task 2 학습할 때, sub network 추가, 기존 NN의 weight 고정(catastrphic forgetting 방지), 기존 NN의 i-th layer의 output은 sub network의 i+1-th layer의 추가 input으로 사용 - 기존 weight를 sub network에 통합하는 과정을 lateral connection(측면 연결)이라 함
       - Task 3 학습할 때, task 2 때와 같이, sub network 추가, 기존 NN의 weight 고정, laternal connection
       - 각 task 학습 이후 새로운 NN(column)을 추가하는 방법으로 catastrophic forgetting 방지, lateral connection으로 knowledge transfer

3. Memory
    - Deep Generative Replay (DGR) by SK T-Brain: 생물학적 기억 mechanism을 모방 <-> regularization, structure는 NN modeling 방식
      - 뇌의 해마를 모방하여 만든 알고리즘
        - 해마는 뇌로 들어온 감각 정보를 단기간 저장하고 있다가 이를 대뇌피질로 보내 장기 기억으로 저장하거나 삭제
        - DGR은 단기 기억과 장기 기억의 상보적 학습 관계를 generator와 solver로 구현
        - Generator는 GAN 기반, 학습했던 데이터와 유사한 데이터를 replay
        - Solver는 주어진 task를 해결하는 장기 기억 역할, new task를 학습할 때 generator가 생성한 old task에 대한 데이터를 동시에 학습
        - 다른 모델에 학습된 지식을 전달하는 것도 가능 - Scholar(학자) model
      - Training
        - DGR은 Task N개를 순차적으로 학습
        - Task 1을 학습할 때, generator는 데이터를 replay하도록 학습, solver는 task 1에 대해 학습, 이 상태를 scholar 1
        - Task 2부터 단기 기억 사용, Task 2의 데이터를 x, 정답을 y라 하고 generator가 task 1을 replay한 것을 y'라 하면, generator는 x와 x'을 동시에 replay하도록 학습
        - x'을 solver에 입력하여 얻은 결과를 y'라 하면, solver는 아직 task 2를 학습하기 전이므로 x'와 y'는 task 1의 데이터를 재현한 것과 그에 대한 예측
        - x, y, x', y'을 모두 사용하여 task 1과 task 2를 동시에 수행할 수 있도록 solver를 학습
        - 새로운 task를 학습하면서 catastrophic forgetting을 막음

4. Fusion
    - Dynamically Expandable Network (DEN) by KAIST
      - Regularization과 structure 접근법을 혼합
        - 최초 task 1에 대한 학습은 L1 regularization을 이용하여 weight가 sparsely 학습
        - Weight가 0일 경우 model에서 해당 weight parameter 삭제
        - 새로운 task 학습
        - Selective retraining 단계, 중요한 node 탐색, weight 갱신
          - Layer 1까지의 weight 고정, L1 regularization 학습, task B 학습에 중요한 layer 2의 node를 찾을 수 있음
          - Node를 따라가면 task B를 학습하는데 중요한 node와 weight를 찾을 수 있음
          - 탐색이 끝나면 L2 regularization으로 학습하며 중요한 node와 weight를 미세 조정
        - Dynamic network expansion 단계, 새로운 task를 학습하는데 모델의 capacity가 부족할 경우 network를 확장
          - Selective retraining 결과, 새로운 task에 대한 loss가 threshold 이상일 경우, 모델의 capacitiy가 부족하다 판단
          - Layer 별로 임의의 개수 만큼 node 추가
          - Group sparsity regularization을 이용해 추가된 k개의 node 중 필요 없는 nodes 제거, 제거되지 않은 node도 sparse하게 함
        - Network split/duplication 단계, sematic drift 확인 및 해소
          - Task B를 학습하기 전 weight와 학습 후 weight 사이의 L2 distance 계산
          - L2 distance가 임계치를 넘으면 semantic drift가 발생했다고 판단, 해당 node를 복제하여 layer에 추가
          - 임계치를 넘은 node가 기존 값을 유지하도록 regularization으로 규제하며 모델이 재학습하도록 함
        - DEN의 학습 과정은 이전 task에 관련된 weight는 최대한 유지하면서 새로운 task를 학습하기 때문에 catastrophic forgetting 예방
        - Progressive netoworks보다 capacity를 효율적으로 

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
- [메타 논문 생성 AI '갤럭티카' 수준미달로 3일만에 폐쇄](http://www.aitimes.com/news/articleView.html?idxno=148034), 2022-11-22-Tue.
- ["품질이 형편없네"... 메타가 만든 AI챗봇, 이대로 괜찮나](http://www.aitimes.com/news/articleView.html?idxno=146343), 2022-11-22-Tue.
- [원하는 텍스트 콘텐츠 생성해주는 앱 '노션 AI'](http://www.aitimes.com/news/articleView.html?idxno=147997), 2022-11-22-Tue.
- [딥마인드, 코드 생성 AI '알파코드' 공개](http://www.aitimes.com/news/articleView.html?idxno=148344)

### Articles unrelated to AI

- [이베이코리아 인수전](https://it.donga.com/31732/), 2021-03-28-Mon.
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
