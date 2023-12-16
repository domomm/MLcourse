import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from itertools import cycle


def plt_Training_Val_Accuracy(history):
    plt.figure(figsize=(4,3))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    
def plt_Training_Val_loss(history):
    plt.figure(figsize=(4,3))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    

def plt_conf_matrix(conf_matrix):
    # Display the confusion matrix using a heatmap
    class_labels = ['normal [0]', 'bacteria [1]', 'virus [2]']

    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
#code from the assignment starts here
def display_xRAY(X):
    fig, ax = plt.subplots(1,1, figsize=(2,2))
    X_reshaped = X.reshape((100,100)).T
    # Display the image
    ax.imshow(X_reshaped, cmap='gray')
    plt.show()
#code from the assignment ends here

def plot_roc_curve(y_true, y_score, num_classes):
    """
    Plot the ROC curve for binary or multiclass classification.

    Parameters:
    - y_true: True labels (ground truth)
    - y_score: Predicted scores (probabilities) for each class
    - num_classes: Number of classes in the classification task (default is 2 for binary)

    Note: For multiclass classification, y_score should contain the predicted probability for each class.
    """
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    y_score_bin = label_binarize(y_score, classes=range(num_classes))

    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score_bin.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curves
    plt.figure(figsize=(8, 6))

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive'])

    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve (area = {:.2f})'.format(roc_auc[i]))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()