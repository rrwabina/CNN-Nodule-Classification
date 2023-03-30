import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from torchvision.utils import make_grid
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

from PIL import Image
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import warnings 
warnings.filterwarnings('ignore')

def epoch_time(start_time, end_time):
    ''' 
    Arguments: 
        start_time 
        end_time   
    Returns the elapsed minute and seconds of model training
    '''
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def get_sampler(dataset):
    '''  
        Arguments:
            dataset: Any imbalanced dataset that requires oversampling
        Returns:
            dataset used with Weighted Random Sampler
    '''
    classes = [label for _,   label in dataset]
    index_0 = [idx   for idx, label in enumerate(classes) if label == 0]
    index_1 = [idx   for idx, label in enumerate(classes) if label == 1]
    weights = torch.zeros(len(index_0) + len(index_1))
    weights[index_0] = 1.0 / len(index_0)
    weights[index_1] = 1.0 / len(index_1)
    sampler = WeightedRandomSampler(weights = weights, num_samples = len(weights), replacement = True)
    return sampler

def count_parameters(model, print_all = True):
    ''' 
        Arguments:
            model: Deep learning to be analyzed in terms of number of parameters
            print_all: Print the number of parameters for each layer
        Returns:
            number of parameters
    '''
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    if print_all:
        for item in params:
            print(f'{item:>8}')
    print(f'________\n{sum(params):>8}')

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def plot_metrics(train_losses, valid_losses, train_accurs, valid_accurs):
    alpha = 0.3
    smoothed_train_losses = [train_losses[0]]
    smoothed_valid_losses = [valid_losses[0]]
    smoothed_train_accurs = [train_accurs[0]]
    smoothed_valid_accurs = [valid_accurs[0]]
    
    for i in range(1, len(train_losses)):
        smoothed_train_losses.append(alpha * train_losses[i] + (1-alpha) * smoothed_train_losses[-1])
        smoothed_valid_losses.append(alpha * valid_losses[i] + (1-alpha) * smoothed_valid_losses[-1])
        smoothed_train_accurs.append(alpha * train_accurs[i] + (1-alpha) * smoothed_train_accurs[-1])
        smoothed_valid_accurs.append(alpha * valid_accurs[i] + (1-alpha) * smoothed_valid_accurs[-1])
    
    smoothed_train_losses = train_losses
    smoothed_train_accurs = train_accurs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5))
    ax1.plot(smoothed_train_losses, label = 'Train')
    ax1.plot(smoothed_valid_losses, label = 'Valid')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Losses')
    ax1.legend()

    ax2.plot(smoothed_train_accurs, label='Train')
    ax2.plot(smoothed_valid_accurs, label='Valid')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracies')
    ax2.legend()
    plt.show()

def display_loader(loader, class_names = ['Benign', 'Malignant'], nrow = 6):
    for images, labels in loader:
        break

    print('Label:', labels.numpy())
    print('Class: ', *np.array([class_names[i] for i in labels]))

    im = make_grid(images, nrow=nrow)
    plt.figure(figsize=(15, 8))
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
    plt.axis('off')
    plt.show()

def check_imbalance(dataset):
    classes = [label for _,   label in dataset]
    index_0 = len([idx   for idx, label in enumerate(classes) if label == 0])
    index_1 = len([idx   for idx, label in enumerate(classes) if label == 1])
    return index_0, index_1 

def plot_imbalance(train_dataset, valid_dataset):
    class_count_train = check_imbalance(train_dataset)
    class_count_valid = check_imbalance(valid_dataset)

    fig, axs = plt.subplots(1, 2, figsize = (12, 6))
    fig.suptitle('Class distribution in Training and Validation sets', size = 15)
    labels = ['0: Benign', '1: Malignant']

    axs[0].bar(labels, class_count_train)
    axs[0].set_title('Train set')
    axs[0].set_xlabel('Class', size = 12)
    axs[0].set_ylabel('Number of samples')

    axs[1].bar(labels, class_count_valid)
    axs[1].set_title('Validation set')
    axs[1].set_xlabel('Class', size = 12)
    axs[1].set_ylabel('Number of samples')
    plt.show()