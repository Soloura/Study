# OpenCV Spatial AI Competition Winners
- Intel이 스폰한 대회로 수상팀 6팀을 다음과 같이 발표함
- 올해 7월에 시작한 대회로 며칠만에 235개 이상의 참가를 받았음
- 첫번째 단계에서 32팀을 선정했고, 수준 높은 결과물로 고르기 어려워 기한을 연장함
- 내년 1월에 10배의 상금으로 대회를 새롭게 열 예정임

1. Vision System for visually impaired ($3000)
- 장인이나 시각적으로 불편함을 가지고 있는 사람들을 돕기 위한 인지 시스템으로, OAK-D 센서를 이용하여 실내, 실외 모두에서 길 안내가 가능하다. 신호, 장애물, 횡단보도, 움직이는 물체, 계단들을 음성으로 안내해준다. Monrovia, California에서 테스트한 결과, 일반적인 상황에서 동작했으며, 현재 연구팀은 논문으로 준비하고 있다.

2. Universal Hand Control ($3000)
- 일상 생활에서 불, TV 등을 끄면서 보내는 시간을 좀 더 생산적인 일을 할 수 있도록, 효율적으로 쓰고 싶어서 시작한 프로젝트로, OAK-D를 이용하여 nerual networks를 이용해 손의 자세를 예측하고 depth를 통해 손의 위치를 측정하여 여러 기기들을 제어하거나 키보드나 마우스를 사용하지 않고 직접 컴퓨터를 제어할 수 있다. 모델을 튜닝하는데 PINTO의 Zoo 레포를 참고하였다.

3. Parcel Classification & Dimensioning ($2000)
- 화물 크기를 측정해서 알맞는 크기의 컨테이너에 포장하는 프로젝트로, DepthAI USB3 (OAK-3) 카메라를 이용하여 모양, 너비, 길이, 높이를 측정하여 계산한다.

4. Real Time Perception for Autonomous Forklifts ($2000)
- 자율 주행 지게차 프로젝트로, 화물을 어떻게 이동시킬지 vision과 geo를 이용했다.
- 지게차의 동작은 창고 안에서 load, unload, transport로 이루어져있다.

5. At Home Workout Assistant ($1000)
- 사회적 거리 두기로 인하여 체육관이나 공공 장소를 나가지 못해 운동을 못해서, 혼자 운동하기 위한 프로젝트 Motion Analysis for Hoe Gym System(MAHGS)이다.
- 혼자 운동하는건 어렵고 근육 부상 등으로 위험할 수도 있어서 집에서 운동하는 걸 도와주는 프로젝트로, 3D로 사람의 자세를 측정하는 것과 움직임을 분석하는 것으로 나뉘어져있다. 

6. Automatic Lawn Mower Navigation ($1000)
- 잔디깍이는 시간도 많이 들고 귀찮은 집안일인데, 이를 자동으로 해주는 것도 줄을 설치해서 영역이 한정된다는 것과 작은 동물을 죽이거나 다치게 할 수 있거나 나무 같은 물체에 충돌하는 등의 한계가 있다.
- Obstacle detection으로 물체들을 피해 이런 문제들을 예방하도록 설계했다.