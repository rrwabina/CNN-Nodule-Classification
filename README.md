# CNN Nodule Classification
In this repository, we utilized Convolutional Neural Networks (CNN) to develop a binary classification model in detecting nodules in CT scans. We implemented all models using PyTorch version 3.9, where the Intel 8th generation CPU performed all simulation in this study witn an NVIDIA RTX 1050Ti 4GB graphics card.

You can access the main Jupyter Notebook in the file:
```
../notebooks/main.ipynb
```
This notebook utilized pre-trained CNN models (i.e., AlexNet, VGG16, and DenseNet). The user may also want to explore other CNN models (not pre-trained), such as LeNet and GoogleNet, written in Pythonn files. Feel free to explore!
```
../architecture
```
## Background
Nodule binary classification is a field of study in medical imaging that aims to differentiate between benign and malignant nodules detected in imaging scans like CT scans. Nodules are small growths that can occur in various organs and accurate classification of nodules is crucial for the diagnosis and treatment of conditions like cancer.

The use of imaging scans for nodule detection has increased in recent years, resulting in a higher number of nodules being identified. However, accurately distinguishing between benign and malignant nodules can be challenging as they can look similar in medical images.

Artificial intelligence and machine learning techniques have shown potential in improving the accuracy of nodule classification by analyzing large amounts of medical images and identifying patterns and features that differentiate between benign and malignant nodules. This area of research is important for early cancer detection and improving patient outcomes.

## Dependencies
You need to install the prerequisites.
``` 
pip install -r requirements.txt
```

## Model Formulation
We created our proposed model in a class called <code>SamuelNet</code>. This model takes an input size of $50 \times 50$ tensors and our output represents the 2 possible digits - either 0 or 1. We used AlexNet architecture as the baseline model for the nodule classification. However, we modified the baseline model in terms of the feature extractions and classifiers. 

We changed the AlexNet's feature extraction layers from five (5) to two (2) convolutions in order to reduce model complexity and avoid overfitting. These convolution layers takes three kernel filters with 1 stride and a single padding. Each convolution is followed by a max-pooling layer. The output of the convolution layers is fed to the ReLU function to avoid overfitting. We added Batch Normalization layers to each convolution layers to help <code>SamuelNet</code> learn faster and reduce the internal covariate shift, to avoid slowing down the training process. The Batch Normalization can also improve the generalization of our network since it can reduce the dependence on the initial weight initialization and the choice of hyperparameters.

Meanwhile, the <code>SamuelNet</code>'s classifier uses three fully-connected (FC) layers. However, we modified the third FC layer from the baseline model with 2 channels to classify 2 classes, instead of the regular 10 classes.

We also added dropouts in our model to improve generalization. 
<center>
<img src="/figures/samuelnet.PNG" width = "808"/>
</center>
Note: <code>SamuelNet</code> is not a published CNN model framework. This is a project as a partial requirement of the author in RADI605: Modern Machine Learning class.

## Summary
The table below shows the summary of performance metrics of different CNN models, along with their number of parameters, testing accuracy, and F1-score for both labels. As seen in the table, **ModifiedSamuelNet** acquired the best performance metrics among the trained models.

Note: The VGG16** was not shown in notebook but was trained separately on Google Colab. The DenseNet-121 (72) refers to the best model acquired during 72nd epoch of model training.

| Model                 |  Parameters    | Testing Accuracy | F1-score (0)    | F1-score (1)    |
|:---------------------:|----------------|------------------|-----------------|-----------------|
| SamuelNet             |   1,666,530    |  84%             | 90%             | 61%             |
| AlexNet               |     120,770    |  50%             | 65%             | 15%             |
| VGG16                 |       8,194    |  50%             | 65%             | 15%             |
| VGG16**               | 139,578,434    |  17%             | 49%             | 14%             |
| DenseNet-121          |   7,978,856    |  82%             | 89%             | 50%             |
| DenseNet-121 (72)     |   7,978,856    |  91%             | 95%             | 67%             |
| ModifiedSamuelNet     |   6,761,986    |  95%             | 97%             | 80%             |
