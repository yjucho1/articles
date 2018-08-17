# Visualizing and Understanding Convolutional Networks

#### Matthew D. Zeiler, Rob Fergus, 2013

* 1990년대 LeCun et al.(LeNet5)이후 컨볼루션 네트워크는 아주 우수한 성능을 내며 발전해왔습니다. 
* 눈부신 발전에도 불구하고 CNN의 내부 오퍼레이션과 변화양상에 대한 통찰과 어떻게 좋은 성능을 얻게 되었는지에 대한 연구는 거의 없었습니다. 
* 이 논분은 Deconvolutional Network(deconvnet)라는 시각화 기법을 사용하여 입력의 각 픽셀들이 피쳐맵에 어떻게 투영되는지(project) 알아봅니다.
* 또한 입력 이미지의 일부분을 가려보고 이것이 아웃풋에 어느정도 영향을 주고, 이미지의 어떤 부분이 최종 분류 판단에 가장 중요한지 알아보기 위해 민감도 분석을 실시하였다. 