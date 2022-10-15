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
The dataset used for this project is the famous CIFAR10 dataset containing a total of 60000 images of 32 by 32 pixel size which is available in the PyTorch framework's CIFAR10 object from torchvision.datasets module.
Once the dataset is created, DataLoaders are also created from the torch.utils.data module for both the train and the test set.

![CIFAR data](https://github.com/SohamBera16/Image-Classification-with-CIFAR-10-dataset/blob/main/cifar10.png)

## Data Exploration: 
The images in the dataset are viewed using the show5 function defined which takes a data loader as an argument. ![cifar sample](https://github.com/SohamBera16/Image-Classification-with-CIFAR-10-dataset/blob/main/cifar10%20data.png)

## Data Preparation
The training dataset containing 50K images have been divided into train and validation sets with a ratio of 9:1. The test dataset contains the rest of the images i.e 10K images.

## Data Augmentation
The training dataset is transformed for increasing the variation using the transforms module from torchvision package. In order to increase the dataset so that the model learns better and generalizes well (instead of overfitting),  various data augmentation techniques like - _Normalization, Rotation, Flipping, and Cropping_ have been introduced to the CIFAR-10 dataset. 

## Model generation:
Using the layers in torch.nn (which has been imported as nn) and the torch.nn.functional module (imported as F), a neural network has been constructed based on the parameters of the dataset.  Various Transfer Learning architectures with pretrained weights like Resnet18 and DenseNet121 were imported for testing as well. The NLL loss function and Adam optimizer were created for the model training. 

## Model Evaluation:
As there is no class imbalance in the training dataset, accuracy has been chosen as the model evaluation metric. 

## Saving the models for future inferences:
All the models, i.e. the handcrafted CNN model, Resnet18, and Densenet121 with fine tuning were saved for future inference applications.

## Final Results:
The pretrained transfer learning models, in particular Densenet121, with fine tuning improved the accuracy metric by 20% even with a training of 20 epochs due to resource limitation. 

## Future Work:
Running the model for more number of epochs, Testing with other models, Performing hyperparameter optimization with a larger search space, etc.

N.B. The project is undergoing and hence the latest updates will be 

