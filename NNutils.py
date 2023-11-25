import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



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