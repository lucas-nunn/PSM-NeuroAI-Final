# CREDIT: ADRIEN DOERIG

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform

def plot_batch_predictions(model, loader, n_imgs):
    # Get a batch of images and labels from the dataloader
    img_batch, label_batch = next(iter(loader))

    # if we only have one color channel, use a grayscal colormap
    # otherwise, use the standard 3-channel one
    if img_batch.shape[1] == 1:
      colormap = 'gray'
    else:
      colormap = None

    # Pass the images through the model to get the predicted labels
    predicted_labels = model(img_batch)

    # Make plot with predicted vs. true labels
    # we'll have max 10 images per row
    n_rows = int(np.ceil(n_imgs/10))
    n_cols = min(10, n_imgs)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols,n_rows*1.5))
    if n_rows == 1:
        # Need to make sure ax is 2D array even when there is a single row
        # Otherwise we get wrong indexing in the for loop below
        ax = np.expand_dims(ax, axis=0)

    for i in range(n_imgs):
        row, col = i//10, i%10
        ax[row, col].imshow(img_batch[i].permute(1,2,0), cmap=colormap)
        ax[row, col].set_title(f'pred={torch.argmax(predicted_labels[i])}\ntrue={label_batch[i]}')
        ax[row, col].set_xticks([])
        ax[row, col].set_yticks([])


def plot_dataset_samples(loader, n_imgs):

    # get data from dataloader
    example_data, example_labels = next(iter(loader))

    # if we only have one color channel, use a grayscal colormap
    # otherwise, use the standard 3-channel one
    if example_data.shape[1] == 1:
      colormap = 'gray'
    else:
      colormap = None

    # Make plot with predicted vs. true labels
    # we'll have max 10 images per row
    n_rows = int(np.ceil(n_imgs/10))
    n_cols = min(10, n_imgs)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols,n_rows*1.5))
    if n_rows == 1:
        ax = np.expand_dims(ax, axis=0)

    for i in range(n_imgs):
        this_img = example_data[i].permute(1,2,0)
        row, col = i//10, i%10
        ax[row, col].imshow(this_img, cmap=colormap)
        ax[row, col].set_title(f"Label: {example_labels[i]}")
        ax[row, col].set_xticks([])
        ax[row, col].set_yticks([])
    plt.show()

def plot_pca(activations, labels, name):
    # Run PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(activations)

    # Plot PCA results
    plt.figure(figsize=(5, 4))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, ticks=range(10))
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f'PCA of Layer {name} Activations')
    plt.show()

def plot_tsne(activations, labels, name):
  # Run t-SNE
  tsne = TSNE(n_components=2, random_state=42)
  tsne_result = tsne.fit_transform(activations)

  # Plot t-SNE results
  plt.figure(figsize=(5,4))
  scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='tab10', alpha=0.7)
  plt.colorbar(scatter, ticks=range(10))
  plt.xlabel('t-SNE Component 1')
  plt.ylabel('t-SNE Component 2')
  plt.title(f't-SNE of Layer {name} Activations')
  plt.show()
  
def plot_rsa(activations, labels, name):
  # Arrange data by label, and run RSA
  sorted_indices = np.argsort(labels)  # Get indices that would sort the labels array

  # Sort both data and labels using the sorted indices
  sorted_activations = activations[sorted_indices]
  sorted_labels = labels[sorted_indices]

  # Calculate pairwise distances using pdist
  pairwise_distances = pdist(sorted_activations, 'correlation')
  plt.figure(figsize=(5,4))
  plt.matshow(squareform(pairwise_distances))
  plt.title(f'RSA for {name}')
  plt.colorbar()
  plt.show()