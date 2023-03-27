import tensorflow as tf
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct       = (rounded_preds == y).sum() 
    acc           = torch.mean(torch.eq(preds, y).float())
    return acc

def accuracy(prediction, targets):
    rounded_prediction = torch.round(torch.sigmoid(prediction))
    correct = (prediction == targets[0 : 3249]).float()
    return (correct.sum()/len(correct)).item()

def get_metrics(prediction, label):
    prediction  = prediction.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    
    tp = np.sum((prediction == 1) & (label == 1))
    tn = np.sum((prediction == 0) & (label == 0))
    fp = np.sum((prediction == 1) & (label == 0))
    fn = np.sum((prediction == 0) & (label == 1))

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    auc = roc_auc_score(label, prediction)
    fpr, tpr, _ = roc_curve(label, prediction)

    return {'sensitivity': sensitivity,
            'specificity': specificity,
            'auc': auc,
            'fpr': fpr,
            'tpr': tpr}
