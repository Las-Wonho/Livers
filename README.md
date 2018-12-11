# Livers

This is GAN(Generative Adversarial Networks) Implementation.

## data

데이터는 모두 크기가 다른 사진입니다. 얼굴사진이며, 128*128이 가장 적당합니다. 약 200장이상이 있으며, 수집방법은 opencv를 이용한 python스크립트입니다.

## GAN

현제 GAN은 WGAN, DCGAN, Vanila GAN이 구현되어있습니다. 구현방식은 tensorflow를 이용하여서 학습을 했습니다. 깃의 기록을 보시면 예쩐에 생성했던 이미지가 있습니다.

## Trainning

학습은 CPU로 진행하였습니다. 또한 여러 클라우드 컴퓨터에서 학습을 진행하였고, 디펜던시는 아나콘다로 해결하였습니다.

## Dependency

- PIL
- Keras
- Tensorflow
- Numpy
- OpenCV