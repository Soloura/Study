# Reinforcement Learning :robot:

## Q Learning

관찰을 액션에 직접적으로 매핑하는 함수를 학습하는 정책 경사 방법과 달리, Q 러닝은 각 상태 내에서의 값을 학습하고자 시도하며 그 상태 내에서 특정 액션을 취한다. 정책 경사 방법과 Q 러닝 접근법은 궁극적으로는 주어진 환경 아래 지능적인 액션을 취하게 한다는 점에서는 동일하지만 액션에 도달하는 수단이 완전히 다르다.

가장 단순하게 구현한다면 Q 러닝은 주어진 환경에서 가능한 모든 상태(행)와 액션(열)에 대한 값의 테이블로 표현된다. 테이블의 각 셀 내에서 우리는 어떤 상태에서 액션을 취했을 때 그것이 얼마나 좋은지 수치화한 값을 학습한다.

Bellman equation을 이용해 Q 테이블을 업데이트한다. Bellman equation은 주어진 액션에 대해 기대되는 장기 보상이, 현재의 액션에서 얻는 즉각적인 보상과 다음 상태에서 취할 최선의 미래의 액션에서 기대되는 보상의 조합과 같다는 내용이다. 이 아이디어에 의해, 미래의 액션에 대해 테이블을 어떻게 업데이트해야 할지 추정할 때 Q 테이블을 재활용할 수 있다.

Q(s, a)* = r + gamma * (max(Q(s', a')) ... [Eq. 1]

[Eq. 1]의 의미는 어떤 상태 s와 액션 a에 대한 최적의 Q 값은 현재의 보상 r, 그리고 다음 상태 s'에 대해 테이블에 의해 기대되는 할인된 gamma 최대 미래 보상의 합으로 표현할 수 있다. 할인 계수를 사용함으로써 현재의 보상에 비해 미래의 가능한 보상이 얼마나 중요한지 비율을 정할 수 있다. 이런 식으로 업데이트함으로써 테이블은 각 상태에서 취해진 각 액션에 대해 기대되는 미래의 보상을 서서히 정확하게 측정해나간다.

## Q Network

게임이나 실제 환경에서 가능한 상태의 수는 사실 무한대 수준이다. 따라서 대부분의 문제는 단순히 테이블로 해결할 수 없다. 즉 상태를 기술하고 테이블 없이도 액션에 대한 Q 값을 도출해낼 다른 방법이 필요하다. 신경망을 함수의 근사 장치로 동작하게 하면 가능한 상태의 수가 아무리 많더라도 이를 벡터로 표현하여 Q 값에 매핑하는 방법을 학습할 수 있다. 테이블을 직접 업데이트하는 대신, 네트워크에서는 역전파와 loss function을 이용해 업데이트 과정을 처리한다. Loss function으로는 제곱합을 사용하는데, 현재 예측한 Q 값과 타깃 Q 값 간의 차이를 계산하여 그 경사(변화도)가 네트워크에 전달되는 방식이다. 이 경우 선택된 액션에 대한 타깃 Q 값은 앞의 Bellman equation에서 계산한 Q 값과 동일하다.

Loss = sum(Q_target - Q_current)^2 ... [Eq. 2]

## Deep Q Network (DQN) | *Human-level control through deep reinforcement learning* | [Nature](https://www.nature.com/articles/nature14236)

Q 네트워크를 DQN으로 만들기 위해서는 다음과 같은 개선이 필요하다.

- 단일 계층 네트워크를 다계층 합성곱 네트워크로 확장
- 경험 리플레이의 구현, 즉 네트워크가 자신의 경험에 저장된 기억을 이용해 스스로 학습
- 제2의 타깃 네트워크를 활용하여 업데이트 시 타깃 Q 값을 계산

### Convolutional Layer

에이전트는 게임 화면의 출력물을 이해할 수 있어야 한다. 각 필셀을 독립적으로 고려하는 대신 convolutional layer를 이용하면 이미지의 특정 지역을 고려하는 동시에 네트워크의 더 높은 레벨로 정보를 전송하여 스크린 상의 사물과의 공간적 관계를 유지할 수 있다.

### Experience Replay

Experience replay란 에이전트의 경험을 저장해두었다가 랜덤하게 경험의 일부를 뽑아서 네트워크를 학습시키는 것이다. 이는 과제를 더 잘 수행할 수 있게 하여 robust 학습을 간으하게 한다. 랜덤하게 뽑는 경험을 유지함으로써 네트워크가 환경 내에서 즉각적인 것만 학습하는 것을 방지하고 다양한 과거 경험으로부터 학습하게 한다. 각 experience는 (state, action reward, next_state_와 같은 tuple로 저장된다. Experience replay 버퍼는 최근의 기억 중 정해진 몇 개를 저장하며, 새로운 경험이 추가됨에 따라 오래된 경험은 제거된다. 학습을 시작할 때가 되면 단순히 버퍼에서 랜덤한 기억 더미를 뽑아 네트워크를 학습시킨다.

### Target Network

학습 과정 중에 제 2의 네트워크, 즉 target network를 활용한다. 제 2의 네트워크를 사용해, 학습 시 모든 액션에 대한 비용을 계산하기 위해 이용되는 타깃 Q 값을 생성한다. 학습의 각 단계에서 Q 네트워크의 값은 변화(shift)하므로, 이 일련의 변화하는 값을 네트워크 값을 조절하는 데에 이용하면 값을 추정하는 것이 통제 불능 상태에 빠지기 쉽기 때문이다. 즉 네트워크가 타깃 Q 값과 예측 Q 값 간의 피드백 루프에 빠지면서 불안정해질 수 있다. 이런 위험을 줄이기 위해 타깃 네트워크의 가중치는 고정하고 Q 네트워크 값은 주기적 또는 천천히 업데이트되도록 한다. 이런 방식으로 학습을 좀 더 안정적으로 진행할 수 있다.

## Double DQN (DDQN) | Deep Reinforcement Learning with Double Q-Learning | [arXiv](https://arxiv.org/abs/1509.06461)
DDQN의 주된 착안점은 DQN이 각 상태에서 잠재적 액션의 Q 값을 종종 과대평가한다는 사실이다. 이때 모든 액션이 언제나 동일하게 과대평가되면 별문제가 아니겠지만, 실제로는 그렇지 않다고 볼만한 이유들이 제시되었다. 어떤 최적화되지 못한 액션이 최적화된 액션보다 주기적으로 높은 Q 값을 가지게 된다면 에이전트가 이상적인 정책을 학습하기는 어렵다. 이 문제를 바로잡기 위해 학습 단계에서 타깃 Q 값을 계산할 때 Q 값들에서 최댓값을 구하는 대신, 제 1네트워크를 이용해 액션을 선택하고 해당 액션에 대한 타킷 Q 값을 타깃 네트워크에서 생성하는 방법이다. 액션 선택과 타깃 Q 값 생성을 분리하면 추정값이 크게 나오는 일을 상당 부분 줄일 수 있으며 더 빠르고 안저적으로 학습을 진행할 수 있다. 

Q_target = r + gamma * Q(s', argmax(Q(s', a, theta), theta') ... [Eq. 3]

## Dueling DQN | Dueling Network Architectures for Deep Reinforcement Learning | [arXiv](https://arxiv.org/abs/1511.06581)

지금까지 본 Q 값은 특정 상태에서 취해진 특정 액션이 얼마나 좋은지의 정도를 나타내는 값이다. 이를 수식으로 표현하면 Q(s, a)이다. 이와 같은 주어진 상태에서의 액션은 2개의 더 근본적인 개념으로 분해될 수 있다. 첫 번째는 가치 함수인 V(s)로서 단순히 어떤 상태가 얼마나 좋은지 수치화한 것을 의미한다. 두 번째는 어드밴티지 함수 A(a)로 이는 다른 액션에 비해 특정 액션을 취하는 것이 얼마나 좋은지를 수치화한 것이다. 즉 Q는 V와 A의 조합으로 생각할 수 있다.

Q(s, a) = V(s) + A(a) ... [Eq. 4]

Dueling DQN은 어드밴티지 함수와 가치 함수를 분리하여 계산하고 마지막 계층에서만 조합하여 하나의 Q 함수로 만들어주는 네트워크이다. 에이전트가 특정 시간에 가치와 어드밴티지 둘 다에 대해 신경 쓰지는 않을 수 있다. 특정 액션과 연결될 필요를 없애면 상태를 더 robust 추정 값을 얻을 수 있다.

### Learning from Human Preferences | [OpenAI](https://openai.com/blog/deep-reinforcement-learning-from-human-preferences/) [arXiv](https://arxiv.org/abs/1706.03741)

Periodically, two video clips of its behavior are given to a human, and the human decides which of the two clips is closest to fulfilling its goal - in this case, a backflip. The AI gradually builds a model of the goal of the task by finding the reward function that best explains the human's judgments. It then uses RL to learn how to achieve that goal. As its behavior improves, it continues to ask for human feedback on trajectory pairs where it's most uncertain about which is better, and further refines its understanding of the goal.

### AlphaCode

---

### Reference
- 강화학습 첫걸음, 아서 줄리아니, 송교석, 한빛미디어
- 파이썬과 케라스로 배우는 강화학습, 이웅원, 양혁렬, 김건우, 이영무, 이의령, 위키북스 
- Learning from Human Preferences, https://openai.com/blog/deep-reinforcement-learning-from-human-preferences/, 2022-12-09-Fri.
- Reinforcment Learning: An Introduction (2nd Ed.), Richard S. Sutton and Andrew G. Barto, The MIT Press, 2014.
