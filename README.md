# Facial Frontal Lateral Classification
人脸正脸侧脸二分类，一类正脸(label: 0)，一类侧脸(label: 1)

## Network
* resnet50

## Implement
* torch

## Use
* train

```shell
python face_classifiy.py
```

* test

```shell
python face_predict.py sample/frontFace_10000.jpg
```

## Log
* tensorboard

```shell
tensorboard --logdir logs --host 0.0.0.0
```

## Metric
* acc: 0.9728
