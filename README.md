# (creating draft) Generative Adversarial Nets
### Ian J. Goodfellow et al., 2014
> https://arxiv.org/pdf/1406.2661.pdf


이 포스팅은 Unsupervised learning 중 generative adversarial network(GAN)에 대해서 설명하는 포스트입니다.

GAN은 입력데이터의 숨은 구조 혹은 패턴을 찾아내어 입력된 데이터와 유사한(그럴듯해보이는) 데이터를 생성할수 있는 모델을 학습시키는 방벙론이다.
2014년 Ian Goodfellow가 처음 제안한 방법으로 2016년 NIPS 학회에서 큰 관심을 받으면서 여러가지 분야에 접목되고 발전하고 있는듯하다.



GAN은 Alexnet, ResNet처럼 특정한 네트워크의 구조를 말하는게 아니라, 입력데이터와 유사한 이미지를 생성하는 모델(Generator라고 하자)을 학습시키는 방법을 제안한 방법론(framework)이다. 

Goodfellow는 게임이론을 이용하여 generator생성하는 문제를 일종의 최적화 문제로 봤다. 예를들어


