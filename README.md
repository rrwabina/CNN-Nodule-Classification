# CNN Nodule Classification
In this repository, we utilized Convolutional Neural Networks (CNN) to develop a binary classification model in detecting nodules in CT scans. 

## Background

## Model Formulation
We created our proposed model in a class called <code>SamuelNet</code>. This model takes an input size of $50 \times 50$ tensors and our output represents the 2 possible digits - either 0 or 1. We used AlexNet architecture as the baseline model for the nodule classification. However, we modified the baseline model in terms of the feature extractions and classifiers. 

We changed the AlexNet's feature extraction layers from five (5) to two (2) convolutions in order to reduce model complexity and avoid overfitting. These convolution layers takes three kernel filters with 1 stride and a single padding. Each convolution is followed by a max-pooling later. The output of the convolution layers is fed to the ReLU function to avoid overfitting. We added Batch Normalization layers to each convolution layers to help <code>SamuelNet</code> learn faster and reduce the internal covariate shift, which can slow down the training process. The Batch Normalization can also improve the generalization of our network since it can reduce the dependence on the initial weight initialization and the choice of hyperparameters.

Meanwhile, the <code>SamuelNet</code>'s classifier uses three fully-connected (FC) layers. However, we modified the third FC layer from the baseline model with 2 channels to classify 2 classes, instead of the regular 10 classes.

We also added dropouts in our model to improve generalization. 
