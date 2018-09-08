### Unsupervised Deep Embedding for Clustering Analysis
#### J. Xie, R. Girshick, A. Farhadi (University of Washington, Facebook AI Reaserch), 2016

#### keras implementation : https://github.com/XifengGuo/DEC-keras


# DEC consists of two phase
<img src='structure.png' width=500></img>

## 1) pretrain : parameter initialization with deep autoencoder
* 오토인코더를 학습한다. 
* 학습된 오토인코더 중 인코더 부분만 사용하여 압축된 latent vector값으로 k- means 클러스터링을 한다. 
* k-means로 계산된 클러스터의 centroid는 이니셜값이다. phase 2 과정으로 최적화시킨다.

## 2) DEC train : parameter optimization to minimize KL divergence between auxiliary target distribution and cluster assignment distribution (soft assignment)
* phase2는 보조 역할을 하는 타겟분포를 이용해 클러스터 할당 정보로부터 아래 목적으로 학습이 된다. 
* 한 클러스터 속하는 샘플들은 더 가깝고, 다른 클러스터간의 거리는 멀어지도록 한다.
* 학습 과정에서 초기의 인코더 파라미터가 업데이트되면서 latent vector도 업데이트된다. 

* 먼저 centroid u<sub>j</sub> (j번째 클러스터의 중심값)과 z<sub>i</sub>(i번째 데이터의 잠재벡터)를 이용해 i번째 데이터가 j번째 클러스터에 속할 확률 q<sub>ij</sub>를 구한다. – soft assignment
<img src='qij.png' width=200></img>

* 두번째로는 소프트 어싸인(q<sub>ij</sub>)과 타겟분포(p<sub>ij</sub>)사이의 거리를 KL divergence를 사용하여 정의한다.
<img src='KLloss.png' width=200></img>

* p<sub>ij</sub>(target distribution)를 구하는 것이 DEC의 퍼포먼스를 결정하는 핵심인데, 3가지 속성을 갖도록 정의하였다. 
    * 클러스터 내 순도(purity)가 높아지도록 한다.  strengthen predictions (imporve cluster purity)
    * 높은 신뢰도를 갖는 할당에 더 가중치를 준다. put more emphasis on data points assigned with high confidence
    * 클러스터 사이즈로 손실함수 값을 정규화한다. 클러스터 사이즈가 클수록 손실함수에 주는 기여도가 커서 전체 피쳐 공간을 왜곡시키는 것을 방지한다. normalize loss constribution of each centroid to prevent large clusters from distorting the hidden feature space 
    
    <img src='pij.png' width=200></img>



* self-training의 관점으로 높은 신뢰도를 갖는 예측을 더 강화하는 방향으로 최적화 되는 것이다.

## implementation (hyper-parameter setting 참고)
* set network dimension to d-500-500-2000-10 for all datasets (d is data-space dimension)
* iniitalize the weight to random numbers from zero-mean gaussian with std = 0.01
* pretrain : epoch 50000, dropout rate = 20%, learning rate = 0.1 (which is divided by 10 every 20000 iterations, weight decay is set to 0.)
* DEC train : epoch 100000 without dropout,  a constant learning rate of 0.01. The convergence threshold is set to tol = 0.1%
* batch size = 256

## 결과
<img src='result.png' width=400></img>

#### MNSIT 클러스터링 결과 시각화 (t-SNE)
<img src='tSNE.png' width=400></img>

#### Performance on Imbalanced Data 
<img src='imbalanceresult.png' width=400></img>

* MNIST데이터에서 0 클래스의 비율을 r<sub>min</sub>만큼 줄여서 실험함. 
* 다른 클래스의 비율은 모두 1임. 
* fairly robust against cluster size variation.

#### Number of Clusters
<img src='clusternumber.png' width=400></img>

<img src='NMI.png' width=200></img>

<img src='G.png' width=200></img>

* 클러스터 사이즈를 바꿔가면서 실험함. 
* 최종 클러스터 수는 NMI, Generalizability가 sharp jump를 보이는 기점을 선정.
