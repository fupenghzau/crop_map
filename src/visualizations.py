import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
import numpy as np
import torch

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    """
    Plot a confusion matrix for classification results.

    Parameters:
    - y_true (array-like): True labels.
    - y_pred (array-like): Predicted labels.
    - title (str): Title for the plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

def plot_training_loss(train_losses, val_losses=None, title='Training Loss Curve'):
    """
    Plot the training loss curve, optionally including validation loss.

    Parameters:
    - train_losses (list): List of training loss values over epochs.
    - val_losses (list, optional): List of validation loss values over epochs.
    - title (str): Title for the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='b')
    if val_losses is not None:
        plt.plot(val_losses, label='Validation Loss', color='r')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def visualize_features(source_features, target_features, title='Feature Distribution (Source vs. Target)'):
    """
    Visualize the distributions of source and target features using t-SNE.

    Parameters:
    - source_features (torch.Tensor): Source feature tensor of shape (batch_size, num_channels, sequence_length).
    - target_features (torch.Tensor): Target feature tensor of shape (batch_size, num_channels, sequence_length).
    - title (str): Title for the plot.
    """
    # Reshape source and target features to 2D: (batch_size, num_channels * sequence_length)
    source_features_reshaped = source_features.view(source_features.size(0), -1).cpu().numpy()  # Shape becomes (batch_size, num_channels * sequence_length)
    target_features_reshaped = target_features.view(target_features.size(0), -1).cpu().numpy()  # Shape becomes (batch_size, num_channels * sequence_length)

    # Concatenate the reshaped features along the first dimension
    combined_features = np.concatenate([source_features_reshaped, target_features_reshaped], axis=0)  # Shape becomes (2 * batch_size, num_channels * sequence_length)

    # Create labels for visualization: 0 for source, 1 for target
    labels = np.array([0] * source_features.size(0) + [1] * target_features.size(0))

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(combined_features)

    # Plot the t-SNE results
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.colorbar(scatter, ticks=[0, 1], label='Domain')
    plt.clim(-0.5, 1.5)
    plt.xticks([])
    plt.yticks([])
    handles, legend_labels = scatter.legend_elements()
    plt.legend(handles=handles, labels=['Source', 'Target'], loc='best')
    plt.show()
