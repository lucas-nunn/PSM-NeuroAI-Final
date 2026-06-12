# CREDIT: ADRIEN DOERIG

import torch
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA

# First we define a function that fits our incremental PCA
def fit_pca(feature_extractor, dataloader, batch_size, device, n_components=50):

    # Define PCA parameters
    pca = IncrementalPCA(n_components, batch_size=batch_size)

    # Fit PCA to batch
    with torch.no_grad():
        for item in tqdm(dataloader, desc='Fitting incremental PCA...'):

            imgs, labels = item
            imgs = imgs.to(device)

            if imgs.shape[0] < n_components:
                # check if our batch size is at least as large as n_components
                # the last batch can be smaller. If it is smaller than n_components,
                # PCA cannot run so we skip
                continue
            # Extract features
            ft = feature_extractor(imgs)
            # Flatten the features
            ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
            # Fit PCA to batch
            pca.partial_fit(ft.detach().cpu().numpy())

    return pca


# Second, we define a function that does the following:
# 1. Check if we have fitted the PCA already (e.g. because this is the second
# time we extract activities). If so, we use the fitted PCA. Otherwise, we call
# the above function to fit the PCA.
# 2. We get each batch in our dataset, get retrieve the corresponding network
# activities (as done before in our get_activations() function), and apply
# our PCA to it. That will reduce the dimensionality of the batch as required.
def get_pca_activities(feature_extractor, dataloader, device, pca=None):

    if pca is None:
        # Fit PCA
        pca = fit_pca(feature_extractor, dataloader, 50, device)

    features = []
    with torch.no_grad():
        for item in tqdm(dataloader, desc='Extracting activities...'):
            imgs, labels = item
            imgs = imgs.to(device)
            # Extract features
            ft = feature_extractor(imgs)
            # Flatten the features
            ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
            # Apply PCA transform
            ft = pca.transform(ft.cpu().detach().numpy())
            features.append(ft)
    return np.vstack(features), pca