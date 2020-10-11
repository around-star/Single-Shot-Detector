# Single-Shot-Detector
Implementation of [Single Shot Detector](https://arxiv.org/pdf/1512.02325.pdf) in tensorflow.

## Files
- ```base_layer.py``` : Contains the vgg16 architecture as the base layer for feature extraction
- ```ssd300.py``` : Contains the layers following the base layer, which output the predictions.
- ```loss.py``` : Contains the loss function of the network.
- ```batch_gen.py``` : Contain functions for parsing the training files and generating the training set.
- ```utils.py``` : Contains encoding and decoding funtions. The encoding function encodes the training label in a format similar to network output. The decoding function turns back the network output in the label format.
- ```train.py``` : Performs the training of the model.
- ```predict.py``` : Tests the trained model.

**Training is done on [PASCAL VOC](https://www.kaggle.com/huanghanchina/pascal-voc-2012) dataset.**
