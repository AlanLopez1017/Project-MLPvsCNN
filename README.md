# Image Classification: MLP vs CNN on CIFAR-10
## Overview
This project presents a comparative study of two deep learning architectures—Multilayer Perceptrons (MLPs) and Convolutional Neural Networks (CNNs)—for the task of image classification using the CIFAR-10 dataset. 
The goal is to evaluate their performance, complexity, and efficiency under controlled experimental conditions, considering the impact of model size, regularization, and data augmentation.

## Dataset
The experiments use the CIFAR-10 dataset, which consists of 60,000 32×32 color images divided into 10 classes (6,000 images per class).\\
The dataset is split into 50,000 training images and 10,000 test images. A validation set of 10,000 images is also created from the training set.

## Architectures

### Multilayer Perceptron (MLP)
* Fully connected layers (input → hidden → output)\\
* Two main configurations:\\
    * Base architecture (~1.7M parameters)\\
    * Reduced architecture (~820K parameters)\\
* Activation: ReLU\\
* Regularization: Dropout (0.0 or 0.3)\\

### Convolutional Neural Network (CNN)
* Three convolutional blocks with Batch Normalization, ReLU, and MaxPooling
* Fully connected layers at the end for classification\\
* Two main configurations:\\
    * Base architecture (~357K parameters)\\
    * Expanded architecture (~824K parameters)\\
* Regularization: Dropout (0.0 or 0.3)\\
