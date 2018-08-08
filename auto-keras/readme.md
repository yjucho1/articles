
# Auto-keras 이용하기

데이터에 적합한 모델을 자동으로 찾아준다는 오토ML 패키지가 나왔다고 하여 간단히 테스트해보았습니다.

* __*공식 문서 : https://autokeras.com/ *__
* __*깃허브 : https://github.com/jhfjhfj1/autokeras *__

### Auto-keras란?
* Auto-Keras는 자동화된 기계 학습 (AutoML)을 위한 오픈 소스 소프트웨어 라이브러리입니다.
* AutoML의 목표는 데이터 과학 또는 기계 학습에 대한 전문적인 배경지식없이 도메인 전문가가 손쉽게 딥러닝 모델을 학습할수 있도록 하는 것입니다.
* Auto-Keras는 __*딥러닝모델의 아키텍처 및 하이퍼 파라미터를 자동으로 검색하는 기능*__ 을 제공합니다.


### Summary

* torch 기반 패키지입니다. keras과 tensorflow 기반이 아니라니..ㅠ 제가 속았습니다.
    * 코드를 살펴보니 현재는 벤치마크 데이터셋을 불러오는 부분에서만 keras를 사용하고 있습니다.
    * auto-keras의 핵심인 searcher가 정확도가 더 높은 모델을 찾아 최적화하는 부분은 torch 기반으로 작성되어 있습니다.

* 모델 탐색 결과를 저장하고, 다시 불러오고, 구조를 변경하여 학습할 수 있음을 확인하였습니다.
    * torch를 잘 안다면.. 모델 학습 시간을 단축시킬수 있겠어요.
    * 아직까진 분류 문제에 한해서요. 

* 딥러닝 전문가가 모델을 학습하는 것보다 더 좋은 성능의 모델 구조를 찾아내고 최적화된 파라미터를 학습할수 있을까요? (in other words, 우리 사장님이 저는 해고할 가능성이 있을까요?)
    * 충분한 capacity(그러니까 GPU나 TPU같은 장비)가 주어져야할 것같아요. 
    * 더 고도화되면.. 향후엔 기계학습 리서처를 고용하는 대신에 GPU 장비를 더 사는게 경제적일지도 모르겠네요. ㅠㅠ 

# 0. 설치
* pip install auto-keras
    * python 3.6버전만 가능해요!
* pip install git+git://github.com:jhfjhfj1/autokeras.git

# 1. 데이터 불러오기

* keras 패키지를 이용해 잘 알려진 벤치마크 데이터를 가지고 있거나, 개인이 가지고 있는 데이터를 이용할수 있습니다.
* 저는 cifar10 데이터를 이용했습니다. 
* cifar10에 대한 주요 딥러닝 모델별 결과는 여기를 참고하세요.
    * https://en.wikipedia.org/wiki/CIFAR-10
    * DenseNet : 5.19 %, Wide ResNet : 4.0 %



```python
from keras.datasets import cifar10
from autokeras.classifier import ImageClassifier

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

```

    Using TensorFlow backend.


# 2. classifer 정의와 학습

* 오토-케라스를 구성하는 요소는 아래와 같습니다.
    * 여러개의 분류기 모델가 있는 이데아 공간. 탐색해야 하는 space (이미지 분류기를 찾고자 해서 classifier인듯) : classifier
    * 이데아 공간을 탐색하고 최적 모델 구조를 찾아가는 옵티마이저 : searcher
    * 이데아 공간의 한 포인트에 대응되는 임의의 구조를 가진 컨볼루셔널 모델 : graph
    * 임의 구조로 입력데이터에 적합하게 학습된 파라미터를 가진 실제 이미지 분류 모델 : model
* 첫번째 할일은 우선 우리가 산책해야할 공간을 정의하는 것입니다. 
* 그리고 clf.fit()을 하면, 공간을 정의하고 이 공간에서 최적 model을 찾도록 searcher가 clf 공간을 막 돌아댕깁니다.
* searcher가 한발자국 걸어가는 것은 아래와 같은 일이 일어나고 있다는 겁니다.
    * 공간 내 한 포인트에 대응되는 그래프를 그리고, 그 그래프를 학습시킵니다. 
    * 이때 모델을 얼마나 학습시킬지, 얼마나 학습시켜보고 이 모델의 평가할지를 결정할수 있습니다. 
        * default 값은 아래와 같습니다. 
            * MAX_ITER_NUM = 200
            * MIN_LOSS_DEC = 1e-4
            * MAX_NO_IMPROVEMENT_NUM = 5
        * https://github.com/jhfjhfj1/autokeras/blob/master/autokeras/utils.py#L102
    * 저는 cpu만 있는 맥북에서 테스트를 하기때문에 각 모델별로 5 epoch만 학습하도록 하였습니다. 
        * epoch별 정확도는 주피터 노트북에 출력되지 않고, 주피터 로그에 출력이 됩니다.    
<img src='epoch_acc.png' width=500></img>

    * 최종적으로는 5 epoch 의 평균 정확도(출력된 Accuracy)가 모델의 평가 지표로 기록됩니다. 


```python
clf = ImageClassifier(verbose=True, path='auto-keras/', searcher_args={'trainer_args':{'max_iter_num':5}})
```


```python
clf.fit(x_train, y_train, time_limit = 2 * 60 * 60)
```

    Initializing search.
    Initialization finished.
    Training model  0
    Saving model.
    Model ID: 0
    Loss: tensor(3.3173)
    Accuracy 82.7872340425532
    Training model  1
    Father ID:  0
    [('to_wider_model', 6, 64)]
    Saving model.
    Model ID: 1
    Loss: tensor(3.0486)
    Accuracy 83.91489361702129
    Training model  2
    Father ID:  1
    [('to_wider_model', 11, 64)]
    Saving model.
    Model ID: 2
    Loss: tensor(3.0436)
    Accuracy 85.14893617021275
    Training model  3
    Father ID:  2
    [('to_wider_model', 6, 128)]
    Saving model.
    Model ID: 3
    Loss: tensor(2.9816)
    Accuracy 84.93617021276596
    Training model  4
    Father ID:  1
    [('to_wider_model', 6, 64)]
    Saving model.
    Model ID: 4
    Loss: tensor(2.9916)
    Accuracy 85.76595744680851
    Training model  5
    Father ID:  3
    [('to_wider_model', 6, 64)]
    Saving model.
    Model ID: 5
    Loss: tensor(2.9328)
    Accuracy 84.34042553191489
    Training model  6
    Father ID:  0
    [('to_wider_model', 1, 64), ('to_wider_model', 6, 64)]
    Saving model.
    Model ID: 6
    Loss: tensor(2.8635)
    Accuracy 84.82978723404256


# 3. clf의 final_fit

* ddddd


```python
clf.final_fit(x_train, y_train, x_test, y_test, retrain=False)
y = clf.evaluate(x_test, y_test)
print(y)
```

    .........
    Epoch 1: loss 14.130000114440918, accuracy 84.69979296066252
    .........
    Epoch 2: loss 13.873464584350586, accuracy 85.30020703933748
    .........
    Epoch 3: loss 14.748814582824707, accuracy 84.36853002070393
    .........
    Epoch 4: loss 13.878631591796875, accuracy 85.44513457556936
    .........
    Epoch 5: loss 13.748845100402832, accuracy 85.6728778467909
    .........
    Epoch 6: loss 13.842489242553711, accuracy 85.46583850931677
    .........
    Epoch 7: loss 14.11633586883545, accuracy 85.13457556935818
    .........
    Epoch 8: loss 13.74773120880127, accuracy 85.23809523809524
    .........
    Epoch 9: loss 14.272468566894531, accuracy 84.38923395445134
    .........
    Epoch 10: loss 14.081094741821289, accuracy 85.40372670807453
    .........
    Epoch 11: loss 14.25650691986084, accuracy 85.32091097308489
    .........
    Epoch 12: loss 13.47047233581543, accuracy 85.81780538302277
    .........
    Epoch 13: loss 13.515745162963867, accuracy 85.21739130434783
    .........
    Epoch 14: loss 14.218124389648438, accuracy 84.4927536231884
    .........
    Epoch 15: loss 13.876603126525879, accuracy 85.15527950310559
    .........
    Epoch 16: loss 17.803131103515625, accuracy 80.26915113871635
    .........
    Epoch 17: loss 13.56763744354248, accuracy 85.83850931677019
    .........
    Epoch 18: loss 15.81383228302002, accuracy 83.87163561076605
    .........
    Epoch 19: loss 14.330272674560547, accuracy 85.09316770186335
    No loss decrease after 5 epochs
    0.8463768115942029



```python
clf.get_best_model_id()
```




    4




```python
clf.path 
## default path is /tmp/autokeras/
## if you want to change path, create clf with path pram.
# clf = ImageClassifier(verbose=True, path='auto-keras-test/')
```




    'auto-keras/'




```python
from keras.models import load_model

model = load_model('4.h5')
```


    ---------------------------------------------------------------------------

    OSError                                   Traceback (most recent call last)

    <ipython-input-10-0dd5d0bb3076> in <module>()
          1 from keras.models import load_model
          2 
    ----> 3 model = load_model('4.h5')
    

    /anaconda3/envs/py36/lib/python3.6/site-packages/keras/engine/saving.py in load_model(filepath, custom_objects, compile)
        247     opened_new_file = not isinstance(filepath, h5py.File)
        248     if opened_new_file:
    --> 249         f = h5py.File(filepath, mode='r')
        250     else:
        251         f = filepath


    /anaconda3/envs/py36/lib/python3.6/site-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, **kwds)
        310             with phil:
        311                 fapl = make_fapl(driver, libver, **kwds)
    --> 312                 fid = make_fid(name, mode, userblock_size, fapl, swmr=swmr)
        313 
        314                 if swmr_support:


    /anaconda3/envs/py36/lib/python3.6/site-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
        140         if swmr and swmr_support:
        141             flags |= h5f.ACC_SWMR_READ
    --> 142         fid = h5f.open(name, flags, fapl=fapl)
        143     elif mode == 'r+':
        144         fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)


    h5py/_objects.pyx in h5py._objects.with_phil.wrapper()


    h5py/_objects.pyx in h5py._objects.with_phil.wrapper()


    h5py/h5f.pyx in h5py.h5f.open()


    OSError: Unable to open file (unable to open file: name = '4.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)



```python
searcher = clf.load_searcher()
searcher.history
```




    [{'model_id': 0, 'loss': tensor(3.3173), 'accuracy': 82.7872340425532},
     {'model_id': 1, 'loss': tensor(3.0486), 'accuracy': 83.91489361702129},
     {'model_id': 2, 'loss': tensor(3.0436), 'accuracy': 85.14893617021275},
     {'model_id': 3, 'loss': tensor(2.9816), 'accuracy': 84.93617021276596},
     {'model_id': 4, 'loss': tensor(2.9916), 'accuracy': 85.76595744680851},
     {'model_id': 5, 'loss': tensor(2.9328), 'accuracy': 84.34042553191489},
     {'model_id': 6, 'loss': tensor(2.8635), 'accuracy': 84.82978723404256}]




```python
graph = searcher.load_best_model()
model = graph.produce_model()
```


```python
model.layers
```




    [ReLU(),
     Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1.5, 1.5)),
     BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
     Dropout2d(p=0.25),
     MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
     ReLU(),
     Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1.5, 1.5)),
     BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
     Dropout2d(p=0.25),
     MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
     ReLU(),
     Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1.5, 1.5)),
     BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
     Dropout2d(p=0.25),
     MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
     TorchFlatten(),
     Linear(in_features=1024, out_features=2, bias=True),
     LogSoftmax()]




```python
model.
```




    {'_backend': <torch.nn.backends.thnn.THNNFunctionBackend at 0x1179a0208>,
     '_parameters': OrderedDict(),
     '_buffers': OrderedDict(),
     '_backward_hooks': OrderedDict(),
     '_forward_hooks': OrderedDict(),
     '_forward_pre_hooks': OrderedDict(),
     '_modules': OrderedDict([('0', ReLU()),
                  ('1',
                   Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1.5, 1.5))),
                  ('2',
                   BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                  ('3', Dropout2d(p=0.25)),
                  ('4',
                   MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
                  ('5', ReLU()),
                  ('6',
                   Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1.5, 1.5))),
                  ('7',
                   BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                  ('8', Dropout2d(p=0.25)),
                  ('9',
                   MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
                  ('10', ReLU()),
                  ('11',
                   Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1.5, 1.5))),
                  ('12',
                   BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
                  ('13', Dropout2d(p=0.25)),
                  ('14',
                   MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
                  ('15', TorchFlatten()),
                  ('16', Linear(in_features=1024, out_features=2, bias=True)),
                  ('17', LogSoftmax())]),
     'training': True,
     'graph': <autokeras.graph.Graph at 0x1256eb780>,
     'layers': [ReLU(),
      Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1.5, 1.5)),
      BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
      Dropout2d(p=0.25),
      MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
      ReLU(),
      Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1.5, 1.5)),
      BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
      Dropout2d(p=0.25),
      MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
      ReLU(),
      Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1.5, 1.5)),
      BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
      Dropout2d(p=0.25),
      MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
      TorchFlatten(),
      Linear(in_features=1024, out_features=2, bias=True),
      LogSoftmax()]}




```python
clf2 = ImageClassifier(verbose=True, path='auto-keras/', searcher_args={'trainer_args':{'max_iter_num':5}})
searcher2 = clf2.load_searcher()
graph2 = searcher2.load_best_model()
model2 = graph2.produce_model()
model2.layers
```




    [ReLU(),
     Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1.5, 1.5)),
     BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
     Dropout2d(p=0.25),
     MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
     ReLU(),
     Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1.5, 1.5)),
     BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
     Dropout2d(p=0.25),
     MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
     ReLU(),
     Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1.5, 1.5)),
     BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
     Dropout2d(p=0.25),
     MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
     TorchFlatten(),
     Linear(in_features=1024, out_features=2, bias=True),
     LogSoftmax()]


