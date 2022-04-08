import numpy as np
import matplotlib.pyplot as plt

def get_correaltion(predictions: np.ndarray, 
                    labels: np.ndarray):
    """Computes confusion matrix for data with two or more classes.
    Matrix with fraction of true positives, ture negatives, false 
    positives and false negatives. 
    """
    _is_numpy = (isinstance(predictions, np.ndarray) 
                & isinstance(labels, np.ndarray))
    if not _is_numpy:
        raise ValueError('Inputs must be numpy arrays.')
    
    _n_labels = predictions.ndim
    if not predictions.ndim > 1:
        raise ValueError('Predictions and labels must be an array of at '
                         'least two dimensions')
    res = np.zeros((_n_labels,
                    _n_labels))
    
    predictions = np.argmax(predictions, axis=1)
    labels = np.argmax(labels, axis=1)
    
    for i_prediction in range(_n_labels):
        for i_label in range(_n_labels):
            count = (predictions == i_prediction) & (labels == i_label)
            res[i_prediction, i_label] = np.sum(count)
    return res
    
    
def plot_confusion_matrix(confusion_matrix, 
                          labels):
    """Function which plots the confusion matrix for the given data
    and labels.
    """
    if len(confusion_matrix) != len(labels):
        raise ValueError('Confusion matrix and labels must have the '
                         'same length!')
    
    fig = plt.figure()
    plt.imshow(confusion_matrix, cmap='Greens')
    plt.xticks(ticks=range(2), labels=labels)
    plt.yticks(ticks=range(2), labels=labels)
    plt.colorbar()
    ind = 0
    for x in range(2):
        for y in range(2):
            plt.text(x=x, y=y, s=confusion_matrix[x, y],
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=14
                    )
    return fig
    