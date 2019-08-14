## Multi-Sample Dropout PyTorch

This repository contains a PyTorch implementation of the `Multi-Sample Dropout` from the paper 

[Multi-Sample Dropout for Accelerated Trainingand Better Generalization](https://arxiv.org/pdf/1905.09788.pdf)

by Hiroshi Inoue.

## model 

Multi-Sample Dropout is a new way to expand the traditional Dropout by using multiple dropout masks for the same mini-batch.

![](./png/dropout.PNG)

## Dependencies

* PyTorch
* torchvision

## Uage

The code in this repository implements Multi-Sample Dropout  training, with example on the CIFAR-10 datasets.

To use Multi-Sample Dropout  use the following command.

```python
python run.py --dropout_num = 8
```

## Example

To experiment th result,we use CIFAR-10 dataset for MiniResNet.

```python
# no dropout
python run.py --dropout_num=0

# sample = 1
python run.py --dropout_num=1

# sample = 8
python run.py --dropout_num=8
```

## Results

Train loss of Multi-Sample Dropout with MiniResNet on CIFAR-10.

![](./png/loss.png)

Valid loss of Multi-Sample Dropout with MiniResNet on CIFAR-10.

![](./png/valid_loss.png)

Valid accuracy of Multi-Sample Dropout with MiniResNet on CIFAR-10.

![](./png/valid_acc.png)
