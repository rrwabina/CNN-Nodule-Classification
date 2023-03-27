# CNN Nodule Classification
In this repository, we utilized Convolutional Neural Networks (CNN) to develop a binary classification model in detecting nodules in CT scans. 

You can access the main Jupyter Notebook in the file:
```
../scripts/main.ipynb
```

## Background
Nodule binary classification is a field of study in medical imaging that aims to differentiate between benign and malignant nodules detected in imaging scans like CT scans. Nodules are small growths that can occur in various organs and accurate classification of nodules is crucial for the diagnosis and treatment of conditions like cancer.

The use of imaging scans for nodule detection has increased in recent years, resulting in a higher number of nodules being identified. However, accurately distinguishing between benign and malignant nodules can be challenging as they can look similar in medical images.

Artificial intelligence and machine learning techniques have shown potential in improving the accuracy of nodule classification by analyzing large amounts of medical images and identifying patterns and features that differentiate between benign and malignant nodules. This area of research is important for early cancer detection and improving patient outcomes.

## Model Formulation
We created our proposed model in a class called <code>SamuelNet</code>. This model takes an input size of $50 \times 50$ tensors and our output represents the 2 possible digits - either 0 or 1. We used AlexNet architecture as the baseline model for the nodule classification. However, we modified the baseline model in terms of the feature extractions and classifiers. 

We changed the AlexNet's feature extraction layers from five (5) to two (2) convolutions in order to reduce model complexity and avoid overfitting. These convolution layers takes three kernel filters with 1 stride and a single padding. Each convolution is followed by a max-pooling later. The output of the convolution layers is fed to the ReLU function to avoid overfitting. We added Batch Normalization layers to each convolution layers to help <code>SamuelNet</code> learn faster and reduce the internal covariate shift, which can slow down the training process. The Batch Normalization can also improve the generalization of our network since it can reduce the dependence on the initial weight initialization and the choice of hyperparameters.

Meanwhile, the <code>SamuelNet</code>'s classifier uses three fully-connected (FC) layers. However, we modified the third FC layer from the baseline model with 2 channels to classify 2 classes, instead of the regular 10 classes.

We also added dropouts in our model to improve generalization. 
