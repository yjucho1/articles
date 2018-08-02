###Variational Auto-Encoder 

#### Auto-Encoding Variational Bayes, https://arxiv.org/abs/1312.6114

* 출처 : 
윤상웅님께서 패스트캠퍼스에서 진행하신 딥러닝 알고리즘 워크샵
전인수님께서 패스트캠퍼스에서 진행하신 generative model camp
http://jaejunyoo.blogspot.com/2017/05/auto-encoding-variational-bayes-vae-3.html
https://ratsgo.github.io/generative%20model/2018/01/27/VAE/
https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py



![generative_models](/generative_models.png)

VAE(Variational AutoEncoder)는 데이터가 생성되는 과정(확률분포)을 학습하는 generative model 중 한가지이다.
GAN과 다른 점은 데이터가 분포되어 있는 공간을 특정 확률분포로 가정하고, 그 확률분포의 unknown 파라미터를 추정한다는 점이다.
VAE 역시 일반적인 AutoEncoder처럼 Encoder와 Decoder 부분으로 이루어져있으나, Encoder의 아웃풋이 단순히 차원축소된 벡터가 아니라, 확률분포의 파라미터라는 점이 다르다.

일반적인 AutoEncoder
![general auto-encoder](/AE.png)


VAE 구조
![variational auto-encoder](/VAE.png)


Encoder는 고양이 사진과 같이 고차원의 데이터를 저차원 공간( latent space)상에서 임의의 벡터 z 로 매핑시키는 역할을 한다.
이 때 latent space 상에서 임의의 vector z는 확률분포 Gaussian probability density 를 따른다고 가정하기 때문에 Encoder는 가우시안 분포의 파라미터인 평균과 분산를 아웃풋으로 내보낸다.
q(z|x)=N(μq(x),Σq(x))

Decoder는 잠재공간에 있는 임의의 벡터를 다시 원래의 고차원 데이터로 복원시키는 역할을 한다. 
p(x|z)=N(x|fμ(z),fσ(z)2×I)

VAE의 목적은 입력데이터가 가진 정보를 최대한 보존하면서 저차원의 잠재공간으로 인코딩시키고, 다시 인코딩된 저차원의 임의의 latent vector를 원래의 입력데이터로 최대한 잘 복원시키는 것이다. 
이 목적을 달성하기 위해 VAE는 최적의 확률분포를 찾아내는 방향으로 학습을 진행한다. 최적 확률분포 결정하는 파라미터(theta)는 아래와 같이 log maximum likehood를 최대화하는 방식으로 계산된다.
![log likelihood](/log_likelihood.png)


하지만 우리는 z에 대해서 아는게 없으므로, (무수히 많은 경우의 수에 대해 모두 계산할수 없으므로) 사실 상 최적값을 바로 구하기 어렵다.
이를 해결하기 위해서 Variational Inference 방식을 사용한다. 계산하기 어려운 P(x|z)를 다루기 쉬운 분포 q(z)로 근사하는 것이다. 
이 과정을 수식으로 나타내면 아래와 같다.
![ELBO](/ELBO.png)

마지막 줄에서 나타나듯이 log maximum likehood 를 최대화 하는 값은 결국 ELBO(evidence lower bound)를 최대화하는 것과 같다
보통 딥러닝 모델은 목적함수로 loss function을 최소화하는 형태를 취하므로, 양 변에 마이너스를 취하여 - log p(xi)를 최소화하는 형태로 loss function을 다시 정리하면 아래와 같다.
![VAE loss function](/VAE_loss.png)

먼저 두번째 항은 KL Divergence Regularizer를 의미한다. 보통의 VAE는 z가 zero-mean Gaussian이라고 가정함으로 정규분포끼리의 KLD는 analytic solution으로 도출 가능하다. 이를 적용하면 두번째 항을 다음과 같이 다시 쓸 수 있다.
(https://arxiv.org/pdf/1312.6114.pdf - appendix B를 보면 나와있다)
![KL Divergence](/KLD.png)

첫번째 항은 reconstruction error를 의미한다. 정확한 기대값을 구하긴 어렵지만, q(z|xi)에서 무수히 많은 값을 샘플링하는 방식으로 근사가능하다.
샘플링하는 과정은 본래 미분 불가능한 연산이고 미분이 불가능하면 backpropagation을 할수 없기때문에 실제로 VAE를 학습하기 위해서 reparameterization trick을 쓴다.
http://jaejunyoo.blogspot.com/2017/05/auto-encoding-variational-bayes-vae-3.html - 여기를 참조하자
어쨌든 인코더의 아웃풋인 확률 분포의 평균과 분산을 이용해 샘플링된 벡터 z를 아래처럼 구하는 것이다. 
![reparameterization trick](/reparam.png)

샘플링된 벡터z를 이용해 기대값을 풀어쓰면 입력데이터와 복원된 데이터간의 cross entropy형태가 된다. 
![cross_entropy](/cross_entropy.png)


reconstruction error부분은 입력데이터의 타입에 따라 다르지만, x를 0~1로 정규화시키면 D차원의 binary vector로 표현가능함으로 binary cross entropy를 이용한다.
BCE 외에 MSE를 사용할수 있고, practically 2가지 방법 모두 잘 작동한다고 한다.
![reconstruct error](/recont_error.png)

수식은 복잡하지만 결론은 간단하고, 구현은 더 간단하다.


```python
from keras.losses import binary_crossentropy
from keras import backend as K

reconstruction_loss = binary_crossentropy(inputs, outputs)
reconstruction_loss *= orginal_dim
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
```
