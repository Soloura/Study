# Lifelong Learning | [Wiki](https://en.wikipedia.org/wiki/Lifelong_learning)

Lifelong learning is the "ongoing, voluntary, and self-motivated" pursuit of knowledge for either personal or professional reasons. It is important for an individual's competitiveness and employability, but also enhances social inclusion, active citizenship, and personal development.

### Learning over a lifetime | [Nature](https://www.nature.com/articles/d41586-022-01962-y)

### Self-Net: Lifelong Learning via Continual Self-Modeling | [Frontiers in Artificial Intelligence](https://www.frontiersin.org/articles/10.3389/frai.2020.00019/full)

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

---

### Reference
- Lifelong Learning Blog KR, https://realblack0.github.io/2020/03/22/lifelong-learning.html, 2022-11-09-Wed.
- Lifelong Learning Wiki, https://en.wikipedia.org/wiki/Lifelong_learning, 2023-03-14-Tue.
- Learning over a lifetime Nature, https://www.nature.com/articles/d41586-022-01962-y, 2023-03-14-Tue.
