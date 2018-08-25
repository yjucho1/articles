
* 이 글은 https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html 를 동의하에 번역한 글입니다.

# From GAN to WGAN

이 포스트는 generative adversarial netowrk (GAN) model에 사용되는 수식과 GAN이 왜 학습하기 어려운지를 설명합니다. Wasserstein GAN은 두 분포간의 거리를 측정하는데 더 향상된(smoooth한) 메트릭를 사용하여 GAN의 학습과정을 개선하였습니다.


Generative adversarial network(GAN)은 이미지나 자연어, 음성과 같은 현실의 다양한 컨텐츠를 생성하는 분야에서 큰 성과를 보여주고 있습니다. generator와 discriminator(a critic) 두 모델이 서로 경쟁하듯 학습되어 동시에 서로의 성능이 올라가는 게임 이론에 근본을 두고 있습니다. 하지만 GAN의 학습이 불안정하거나 실패로 이어지는 경우가 많아, 최적에 수렴된 모델을 학습하는 것은 어려운 문제입니다. 

여기서는 GAN에 사용되는 수식들을 설명하고자 하며, 왜 학습이 어려운지, 그리고 학습의 어려움을 해결하기 위해 향상된 GAN을 소개하고자 합니다. 

### Kullback–Leibler and Jensen–Shannon Divergence

GAN을 자세히 설명하기 전에 두 분포사이의 유사도를 정량화하는 두 가지 메트릭을 살펴보도록 하겠습니다. 

1) [KL(Kullback - Leibler) divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) 는 <i>p</i> 분포가 다른 분포 <i>q</i>와 얼마나 떨어져 있는지를 측정합니다. 
<img src ='KL_divergence.gif'></img>
D<sub>KL</sub>는 p(x)==q(x)일때 최소값 zero 입니다.
KL divergence는 비대칭적인 형태라는 점을 기억해두시길 바랍니다. 또한 p(x)가 0에 가깝고 q(x)가 non-zero일 경우, q의 효과는 무시됩니다. 이로 인해 두 분포를 동등하게 사용하여 유사도를 측정하고자 할때 잘못된 결과를 얻을수 있습니다.

2) [Jensen-Shannon Divergence](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence) 는 두 분포의 유사도를 특정하는 다른 메트릭으로 [0, 1] 사이값을 갖습니다. JS divergence는 대칭적입니다(!) 그리고 더 스무스(smooth)하지요. KL divergence와 JS divergence를 더 자세히 비교하는 내용은 이 [Quora post](https://www.quora.com/Why-isnt-the-Jensen-Shannon-divergence-used-more-often-than-the-Kullback-Leibler-since-JS-is-symmetric-thus-possibly-a-better-indicator-of-distance)를 참고하세요.

<img src ='JS_divergence.gif'></img>

<img src ='KL_JS_divergence.png'></img>
<i>Fig.1. 두 가우시안 분포, p는 평균 0과 분산 1이고 q는 평균 1과 분산 1. 두 분포의 평균은 m=(p+q)/2. KL divergence는 비대칭적이지만 JS divergence는 대칭적이다. </i>

### Generative Adversarial Network (GAN)

GAN은 두 모델로 이루어져있습니다.

* discriminator D는 주어진 샘플이 실제 데이터셋에서 나왔을 확률을 추정합니다. 감별사 역할로 실제 샘플과 가짜 샘플을 구분하도록 최적화되어 있습니다.
* generator G는 노이즈 변수인 z(z는 잠재적으로 출력의 다양성을 나타냅니다)를 입력으로 받아 위조된 샘플을 만듭니다. 실제 데이터의 분포를 모사하도록 학습되어 생성한 샘플은 실제 데이터의 샘플과 유사하며, discriminator를 속이는 역할을 합니다.
<img src ='GAN.png'></img>
<i>Fig.2. GAN의 구조 (출처 : [여기](https://www.kdnuggets.com/2017/01/generative-adversarial-networks-hot-topic-machine-learning.html))</i>

학습과정에서 두 모델은 경쟁구조에 놓여 있습니다: G는 D를 속이려고 하고, 동시에 D는 속지 않으려고 합니다. zero-sum 게임에서 두 모델은 각자의 기능을 최대로 향상시킴으로써 서로의 목적을 달성하게 됩니다. 

| Symbol | Meaning | Notes
---------|---------|----
p<sub>z</sub> | 노이즈 입력값 z의 분포 | 보통, uniform|
p<sub>g</sub> | data x에 대한 generator의 분포| |
p<sub>r</sub> | 실제 샘플 x에 대한 데이터 분포 | |
---

우리는 실제 데이터에서 뽑힌 샘플 x의 D의 감별 확률이 높기를 원합니다. 반면에 생성된 샘플 G(z)에 대해서는, z ~ p<sub>z</sub>(z), D의 감별결과값 D(G(z))이 zero에 가깝기를 원합니다. 즉, E<sub>x ~ p<sub>r</sub>(x)</sub>[logD(x)]와 E<sub>z ~ p<sub>z</sub>(z)</sub>[log(1-D(G(z))]가 최대화되길 원합니다. 

하지만 G는 위조된 샘플이 D가 높은 확률로 진짜 데이터라고 판단하도록 학습됩니다. 그래서 E<sub>z ~ p<sub>z</sub>(z)</sub>[log(1-D(G(z))]가 최소화되길 원합니다.

이 두 가지를 합쳐서, D와 G는 minmax game을 하게 되고 아래와 같은 손실함수를 최적화하도록 설계되어 있습니다. 
<img src='GAN_loss.gif'></img>
(E<sub>x ~ p<sub>r</sub>(x)</sub>[logD(x)]는 그래디언트 디센트 업데이트에서 G에 아무런 영향을 주지 않습니다.)

### What is the optimal value for D?

### what is the global optimal?

### what does the loss function represent?

### problems in GANS
