# Image-Classification-with-CIFAR-10-dataset
creating different data pipelines along with a Neural Network model for object detection in the CIFAR-10 dataset and comparing the model performance with a baseline model performance 

## Project requirements and configuration used:
1) Python version 3.6 or higher (used: Python 3.7.14)
2) PyTorch version 1.4 or higher 
3) Torchvision version 0.8 or higher 
4) NumPy version 1.16 or higher 
5) Matplotlib version 3.0 or higher 

## Project Objective: 
To achieve a detection accuracy of greater than 70% for 10 different object categories in the CIFAR-10 image dataset by using various deep learning technologies based on neural networks e.g. dense neural networks, Convolutional neural networks, Transfer Learning, etc.

## Data collection:
The dataset used for this project is the famous CIFAR10 dataset containing a total of 60000 images of 32 by 32 pixel size which is available in the PyTorch framework's CIFAR10 object from torchvision.datasets module. The training dataset is transformed for increasing the variation using the transforms module from torchvision package.
Once the dataset is created, DataLoaders are also created from the torch.utils.data module for both the train and the test set.

## Data Exploration: 
The images in the dataset are viewed using the show5 function defined which takes a data loader as an argument.

## Model generation:
Using the layers in torch.nn (which has been imported as nn) and the torch.nn.functional module (imported as F), a neural network based on the parameters of the dataset.

