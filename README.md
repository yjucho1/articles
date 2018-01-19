# (creating draft) Generative Adversarial Nets
### Ian J. Goodfellow et al., 2014
> https://arxiv.org/pdf/1406.2661.pdf


#### 초록
이 논문에서는 적대적 프로세스(adversarial process)를 통해 생성모델을 추정하는 새로운 방법론을 제안하였다. 이 방법론은 G라는 훈련 데이터의 분포를 추정하는 생성모델과 어떤 샘플데이터가 생성모델 G가 아니라 훈련데이터로부터 뽑힐 확률을 계산하는 판별모델(D)를 동시에 학습하는 것이다. G(생성모델)을 학습하는  과정은 판별모델 D가 잘못 판단할 확률을 높이는 것이다. 이 방법은  two player minimax game과 대응된다. 임의의 G와 D 공간에서, 훈련데이터의 분포를 커버하는 G가 존재하며 D가 1/2이 되는 유일한 해가 항상 존재한다. G와 D가 다중퍼셉트론 모형으로 구성될 경우, 전체 과정은 역전파(backpropagation)를 통해 학습할 수 있다. 모델을 훈련하거나 샘플을 생성하는 과정에서 마코프체인이나 unrolled approximate inference는 사용되지 않는다. 이 논문에서 실험을 통해 생성된 샘플데이터의 정량적, 정성적 평가를 시행하였고 이 방법론의 잠재가능성을 설명하였다.

#### 1. 서론
딥러닝은 이미지 인식, 음성인식, 자연어처리 등 인공지능 산업에서 데이터의 확률분포를 표현하는 다양한 모델을 발견하는 성과를 얻었다. 지금까지 딥러닝에서 가장 큰 성공은 고차원의 센서 입력값을 특정 클래스로 맵핑하는 판별모델(discriminative models)를 학습한 것이다. 이러한 성공적인 모델들은 그래디언트가 잘 작동하는 선형조각을 이용해 역전파와 dropout 알고리즘에 기반한다. 반면 깊은 생성모델은 상대적으로 큰 영향력을 미치지 못했는데, 최우추정 및 관련 방법에서 난해한 확률계산을 근사하기 어렵고 분포를 생성하는 목적에서 선형조각들의 장점을 이용하기 어렵기 때문이다. 이 논문에서 이러한 문제점을 피하면서 새로운 생성모델을 추정하는 과정을 제안하였다. 

이 방식에서 생성모델은 적대적인 과정을 통해 학습된다. 어떤 샘플데이터가 훈련데이터에서 뽑힌것인지 생성모델에서 뽑힌것인지 결정하는 것을 학습하는 판별모델과 적대관계를 가진다. 생성모델은 탐지에 걸리지 않고 위조지폐를 만들고 사용하는 위조지폐범과 유사하고, 반면 판별모델은 위조화폐를 발견하려고 하는 경찰에 비유할수 있다. 이 게임의 경쟁구도를 통해 두 팀은 위조화폐가 진짜와 구분되지 않을 때까지 그들의 방법을 개선한다. 

이 방법론을 이용해 다양한 모델을 학습시키는 알고리즘과 최적화 알고리즘을 만들어낼 수 있다. 이 논문에서는 생성모델과 판별모델 모두 다중퍼셉트론 모형을 이용한 케이스를 탐색했다. 이 특별한 케이스를 적대적 네트워크(adversarial nets)라고 부르도록 하겠다. 이 경우에는 forward propagation을 통해 생성모델로부터 샘플데이터과, backpropagation와 dropout알고리즘을 이용해 두 모델을 학습시킬수 있다. 근사적 추정이나 마코프체인은 필요하지 않다. 

#### 2. 관련문헌

#### 3. 적대적 네트워크
적대적 모델 방법론은 두 모델이 모두 다중레이어모델일때 가장 직접적으로 적용된다. Pg 분포를 학습하기 위해 우리는 노이즈변수에 대한 사전분포 pz를 정의한다. 그리고 pz는 theta_g를 파라미터로하는 다중퍼셉트론으로 표현되는 미분가능한 함수 G(z; theta_g)를 통해 데이터와 맵핑된다. 또한 스칼라 값을 아웃풋으로 하는 두번째 다중퍼셉트론 모델 D(x; theta_d)를 정의한다. D(x)는 x가 p_g가 아니라 데이터로부터 추출될 확률을 나타낸다. 우리는 학습데이터의 샘플과 G로 생성된 샘플 모두 정확히 분류할 확률을 최대화하도록 D를 학습힌다. 동시에 log(1-D(G(z)))를 최소화하도록 G를 학습시킨다. 
즉 D와 G는 V(G, D)함수값을 가지는 minmax문제이다. 

![식(1)](/eq_1.png)

다음 섹션에서는 대다수 네트워크에 대한 이론적 분석을 알아볼 것이다. 훈련 기준은 G와 D가 충분한 데이터와 학습시간이 주어진다면 데이터의 생성분포를 복수할 수 있게 한다 것이다. 직관적인 이해를 위해 그림1을 참조해라. 실제 구현에서는 반복적인 수치 계산을 통해 게임 모델을 학습해야한다. 학습과정의 내부 루프에서 D를 계속 최적화시키는 것은 한정적인 데이터에서 오버피팅이 발생할 수 있게 때문에 제한된다. 대신에 우리는 D를 k단계의 최적화과정과 G를 최적화하는 1단계를 수행한다. 이를 통해 G가 충분히 천천히 변화하면서, D가 최적 해에 가깝게 유지되도록 한다. 이 전략은 SML/PCD 학습과정에서 마코프체인이 무한루프가 되는 것을 피하기 위해 이전단계에서 다음단계로 넘어갈때 마코프체인에서 뽑힌 샘플을 유지하는 것과 유사하다. 이 과정을 알고리즘1로 표현됩니다. 

실제 적용에서, 목적식 1은 G가 충분히 학습되는데 충분한 그래디언트를 제공하지 못한다. 학습초기에는 G의 성능이 좋지 않으므로 D가 생성된 샘플를 구분할 가능성이 높다. 이경우, log(1-D(G(z))) saturate 된다. log(1-D(G(z))) 를 최소화하도록 G를 학습하는 대신에 logD(G(z))를 최대화하도록 G를 학습시킨다. 이렇게 하면 더 큰 그래디언트를 가지면서 동일한 결과를 얻을 수 있다. 

![그림1](/fig_1.png) 
그림1. Generative adversarial nets are trained by simultaneously updating the discriminative distribution
(D, blue, dashed line) so that it discriminates between samples from the data generating distribution (black,
dotted line) px from those of the generative distribution pg (G) (green, solid line). The lower horizontal line is
the domain from which z is sampled, in this case uniformly. The horizontal line above is part of the domain
of x. The upward arrows show how the mapping x = G(z) imposes the non-uniform distribution pg on
transformed samples. G contracts in regions of high density and expands in regions of low density of pg. (a)
Consider an adversarial pair near convergence: pg is similar to pdata and D is a partially accurate classifier.
(b) In the inner loop of the algorithm D is trained to discriminate samples from data, 
converging to D∗(x) = pdata(x)/pdata(x)+pg(x). (c) After an update to G, gradient of D has guided G(z) to flow to regions that are more likely to be classified as data. (d) After several steps of training, if G and D have enough capacity, they will reach a
point at which both cannot improve because pg = pdata. The discriminator is unable to differentiate between
the two distributions, i.e. D(x) = 1/2

 
#### 4. 이론적 결과
생성모델 G는 p_z를 따른 z에 대해 G(z)의 샘플의 분포로 확률분포 p_g를 암시적으로 정의한다.  따라서 충분한 데이터와 학습과정을 통해 알고리즘1이 p_data의 좋은 추정치로 수렴해야한다. 이 섹션의 결과는 비모수적 접근이다. 즉 확률분포함수의 공간에서 수렴성을 통해 inifite capacity를 가진 모델을 표현한다.  

섹션 4.1에서는 이 미니맥스게임문제가 p_g = p_data인 글로벌 최적해를 가진다는 것을 보일것이다. 섹션 4.2에서는 알고리즘1이 목적식1을 최적화시키고, 원하는 결과를 얻는 다는 것을 보일것이다. 

##### 4.1 p_g = p_data에서의 글로벌 최적해
우리는 먼저 G가 주어졌을때 최적 판별기 D에 대해서 생각해보자

프로포지션1. 주어진 G에 대해서 최적 D는
p_data(x) / (p_data(x) + p_g(x))



# well begin is half done!!

참고 reference

* https://brunch.co.kr/@kakao-it/145
* https://www.slideshare.net/NaverEngineering/1-gangenerative-adversarial-network
* https://www.slideshare.net/carpedm20/pycon-korea-2016

