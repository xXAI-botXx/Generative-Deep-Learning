"""
Helper for your latent space!

Use me:

>>> import sys
>>> sys.path += ["./src/helper"]
>>> import latent_space as ls
>>> ls.plot(latent_space:np.ndarray, reduction_methods=["PCA", "TSNE", "UMAP"]
            labels=None, figsize=(18, 5), cmap="viridis"):

Or copy my code to colab 🥳🚀
"""



###############
### imports ###
###############
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

import optuna
import umap    # UMAP uses graphs to reduces the dimensions



###########################################
### multiple laten space visualizations ###
###########################################
def reduce_to_2_dims(latent_space:np.ndarray, reduction_methods=["PCA", "TSNE", "UMAP"], fill_return_with_none=False):
    """
    This function uses hyperparamter optimization to reduce the latent sace to 2 dimensions.
    Dimension reduction methods: PCA, t-SNE and UMAP.

    Please notice following for interpretation:
    - PCA for global informations (distances from clusters to each other)
    - t-SNE for local informations (distances from each point to his neighbors but not from the clusters!)
    - UMAP for global and local informations
    """
    
    if len(reduction_methods) < 1 or len(reduction_methods) > 3:
        raise ValueError(f"Parameter 'reduction_methods' must have 1 to 3 str items!")

    if any[rms not in ["PCA", "TSNE", "UMAP"] for rms in reduction_methods]:
        raise ValueError(f"Found an not allowed value in 'reduction_methods'. Expected 1-3 values from: ['PCA', 'TSNE', 'UMAP'] got: {reduction_methods}")
    
    # Optimizationfunction for Optuna (for t-SNE & UMAP)
    def objective_tsne(trial):
        perplexity = trial.suggest_int("perplexity", 5, 50)
        n_iter = trial.suggest_int("n_iter", 250, 2000)
        
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
        X_embedded = tsne.fit_transform(X)
        
        return silhouette_score(X_embedded, labels)  # Cluster-Qualität messen

    def objective_umap(trial):
        n_neighbors = trial.suggest_int("n_neighbors", 5, 50)
        min_dist = trial.suggest_float("min_dist", 0.0, 0.99)
        
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42)
        X_embedded = reducer.fit_transform(X)
        
        return silhouette_score(X_embedded, labels)

    # hyperparameter optimization
    if "TSNE" in reduction_methods:
        print("Optimizing t-SNE hyperparameters...")
        study_tsne = optuna.create_study(direction="maximize")
        study_tsne.optimize(objective_tsne, n_trials=30)
        best_tsne_params = study_tsne.best_params

    if "UMAP" in reduction_methods:
        print("Optimizing UMAP hyperparameters...")
        study_umap = optuna.create_study(direction="maximize")
        study_umap.optimize(objective_umap, n_trials=30)
        best_umap_params = study_umap.best_params

    # reduction to 2 dimensions
    return_values = None

    # PCA
    if "PCA" in reduction_methods:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        return_values = X_pca
    elif fill_return_with_none:
        return_values = [None]

    # t-SNE with best parameters
    if "TSNE" in reduction_methods:
        tsne = TSNE(n_components=2, perplexity=best_tsne_params['perplexity'], 
                    n_iter=best_tsne_params['n_iter'], random_state=42)
        X_tsne = tsne.fit_transform(X)
        return_values = X_tsne if return_values is None else return_values + [X_tsne] 
    elif fill_return_with_none:
        return_values += [None]

    # UMAP with best parameters
    if "UMAP" in reduction_methods:
        umap_reducer = umap.UMAP(n_neighbors=best_umap_params['n_neighbors'], 
                                min_dist=best_umap_params['min_dist'], 
                                n_components=2, random_state=42)
        X_umap = umap_reducer.fit_transform(X)
        return_values = X_umap if return_values is None else return_values + [X_umap] 
    elif fill_return_with_none:
        return_values += [None]

    return return_values

def plot(latent_space:np.ndarray, reduction_methods=["PCA", "TSNE", "UMAP"]
                        labels=None, figsize=(18, 5), cmap="viridis"):
    """
    This function uses hyperparamter optimization to reduce the latent sace to 2 dimensions.
    Then visualizing the latent space with PCA, t-SNE and UMAP.

    Please notice following for interpretation:
    - PCA for global informations (distances from clusters to each other)
    - t-SNE for local informations (distances from each point to his neighbors but not from the clusters!)
    - UMAP for global and local informations

    Example cmap values:
    - None
    - viridis: Eine der beliebtesten Colormaps, die von dunkelblau bis gelb reicht.
    - plasma: Reicht von dunkelviolett bis gelb-orange.
    - inferno: Reicht von dunkelviolett bis gelb.
    - magma: Reicht von dunkelviolett bis hellorange.
    - cividis: Eine farbenblinde freundliche Variante von viridis.
    - Greys: Graustufen.
    - Blues: Blautöne.
    - Reds: Rottöne.
    - coolwarm: Verläuft von blau über weiß nach rot.
    - hsv: Farbton-Sättigung-Wert (HSV) Farbmodell.
    - jet: Traditionelle Colormap, die von blau über grün nach rot verläuft.
    """
    if len(reduction_methods) < 1 or len(reduction_methods) > 3:
        raise ValueError(f"Parameter 'reduction_methods' must have 1 to 3 str items!")

    if any[rms not in ["PCA", "TSNE", "UMAP"] for rms in reduction_methods]:
        raise ValueError(f"Found an not allowed value in 'reduction_methods'. Expected 1-3 values from: ['PCA', 'TSNE', 'UMAP'] got: {reduction_methods}")

    if not labels:
        labels = "blue"

    # reduce to 2 dimensions
    X_pca, X_tsne, X_umap = reduce_to_2_dims(latent_space=latent_space, reduction_methods=reduction_methods, fill_return_with_none=True):

    # visualization
    original_style = plt.rcParams.copy()
    plt_style = 'seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else np.random.choice(plt.style.available)
    plt.style.use(plt_style)
    print(f"Using '{plt_style}'' plotting style.")

    fig, axs = plt.subplots(1, len(reduction_methods), figsize=figsize)

    idx = 0

    if "PCA" in reduction_methods:
        scatter = axs[idx].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap=cmap, alpha=0.7, edgecolors='k', s=50)
        axs[idx].set_title("Latent Space (reduced with: PCA)")
        axs[idx].xlabel("PCA Dim 1")
        axs[idx].ylabel("PCA Dim 2")
        if type(labels) not str:
            axs[idx].colorbar(scatter, label="True Labels") 
        idx += 1

    if "TSNE" in reduction_methods:
        scatter = axs[idx].scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap=cmap, alpha=0.7, edgecolors='k', s=50)
        axs[idx].set_title("Latent Space (reduced with: t-SNE)")
        axs[idx].xlabel("t-SNE Dim 1")
        axs[idx].ylabel("t-SNE Dim 2")
        if type(labels) not str:
            axs[idx].colorbar(scatter, label="True Labels")
        idx += 1

    if "UMAP" in reduction_methods:
        scatter = axs[idx].scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap=cmap, alpha=0.7, edgecolors='k', s=50)
        axs[idx].set_title("Latent Space (reduced with: UMAP)")
        axs[idx].xlabel("UMAP Dim 1")
        axs[idx].ylabel("UMAP Dim 2")
        if type(labels) not str:
            axs[idx].colorbar(scatter, label="True Labels") 

    plt.show()

    # reset to original plt style
    plt.rcParams.update(original_style)



def original_latent_space_decode_plot():
    """
    FIXME

    -> see in autoencoder notebook for the function
    """
    pass



def interpolation_plot():
    """
    FIXME

    -> see in autoencoder notebook for the function
    """
    pass



def decode_plot():
    """
    FIXME

    -> see in autoencoder notebook for the function
    """
    pass






