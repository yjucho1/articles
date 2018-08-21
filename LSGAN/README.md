# Least Squares Generative Adversarial Networks

#### Xudong Mao et al., 2016

### Introduction

* GAN의 여러가지 발전에도 불구하고, GAN에 의해 보다 괜찮은 데이터들을 생성하는 것은 아직도 몇가지 사례로 제한되어 있다. 이 논문은 Regular GAN이 sigmoid cross entropy loss function을 사용하기 때문에 생기는 문제점을 설명하고 이를 해결하기 위해 least squared loss function 사용하는 것을 제안하였다. 


* regular GAN이 사용하는 loss function은 아래와 같다.
<img src='gan.png' width="500"></img>

* 이 논문에서 제안하는 least squared loss function(LSGAN)은 아래와 같다.
<img src='lsgan.png' width="500"></img>

<img src='fig1.png' width="500"></img>

* Figure 1은 regular GAN이 가지고 있는 문제점을 나타낸다. 
* (b)에서 나타나듯 sigmoid cross entropy loss function은 generator가 생성한 fake samples에 대해서 아주 작은 에러로 계산된다(decision boundary의 아래영역이라서 discriminator가 real sample로 판단한다). 따라서 G를 업데이트 할때 아무런 가르침을 주지 못한다(gradient varnishing).
* (c)는 least squared loss function의 decision boundary이다. generator가 생성한 fake samples은 decision boundary로부터 멀리 떨어져 있고, 이로 인해 패널티를 받는다. 결과적으로 generator는 패널티를 줄이기 위해 decision boundary와 가까운 fake image를 생성하는 방향으로 업데이트된다. 

* contributions
	* discriminator를 학습시키기 위해 least squares loss function을 사용하였다. LSGAN의 목적 함수를 최소화하는 것은 Pearson χ<sup>2</sup> divergence를 최소화하는 것과 같음을 보였다. LSGAN과 regular GAN을 비교하는 실험 결과, LSGAN이 더 우수한 성능을 보였다. 
	* LSGAN의 두 가지 구조를 설계하였다. 첫번째는 112 x 112 이미지 생성을 위한 것으로 여러 종류의 scene datasets에 대해서 우수한 성능을 보였다. 두번째는 수 많은 클래스가 존재하는 문제를 위한 것으로 3470개의 클래스가 있는 handwritten Chinese charactor dataset으로 평가한 결과, LSGAN이 현실적인 수준의 문자 이미지를 생성하는 것을 확인하였다. 

### Related Work
