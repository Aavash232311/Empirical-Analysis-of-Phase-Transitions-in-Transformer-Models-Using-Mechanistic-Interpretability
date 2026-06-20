''' Let's load the model with skeleton in same configuration '''
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

''' For extreme reuseability make those hyperparams also paramaterised '''


def explained_variance(full_path, model, pc):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ''' Important note we can't import the original transfoemr cause there, we have different hyperparamaster initlized. '''

    if full_path is not None:
        try:
            model.load_state_dict(torch.load(full_path, map_location=device)) # loaded into transformer
        except Exception:
            # In later models we have tracked training and eval history for plot
            model.load_state_dict(torch.load(full_path, map_location=device)['model_state_dict'])

    embedding_learned = model.token_embed # token embedding is vocab_size X d_model
    B, T = embedding_learned.weight.shape

    embedding_matrix = embedding_learned.weight.detach()

    X = embedding_matrix.cpu().numpy() 

    pca = PCA(n_components=pc) 

    pca.fit(X)


    explained_variance_ratio = pca.explained_variance_ratio_

    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    plt.figure(figsize=(8, 8))
    plt_x = range(1, len(cumulative_variance_ratio) + 1)
    plt.plot(plt_x, cumulative_variance_ratio, marker='o')
    for i, val in enumerate(cumulative_variance_ratio):
        plt.annotate(f'PC{i+1} {val:.3%} ', 
                    (i, val), 
                    textcoords="offset points", 
                    xytext=(0,8), 
                    ha='center')

    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Explained Variance Ratio by Principal Components')

    plt.show()


def overfitting_plot(train_history, eval_losses):
    epochs = [x['epoch'] for x in train_history]
    train_losses = [x['loss'] for x in train_history]
    
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, train_losses, label='Training Loss', color='red', linewidth=2)
    plt.plot(epochs, eval_losses, label='Evaluation Loss', color='orange', linewidth=2)

    plt.title('Overfitting Check: Training vs Evaluation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()



def pca_analysis(model, pc):
    embedding_learned = model.token_embed
    B, T = embedding_learned.weight.shape
    embedding_matrix = embedding_learned.weight.detach()
    X = embedding_matrix.cpu().numpy()

    pca = PCA(n_components=pc)
    X_pca = pca.fit_transform(X)

    for i in range(len(X_pca)):
        plt.annotate(str(i + 1), (X_pca[i, 0], X_pca[i, 1]), fontsize=8)


    plt.scatter(X_pca[:,0], X_pca[:,1], s=5)
    plt.title("PCA of Embedding vectors model learned")
    plt.show()

def plot_3d(full_path, model, pc):


    if not os.path.exists(full_path):
        sys.exit()  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model.load_state_dict(torch.load(full_path, map_location=device)) 
    except Exception:
        model.load_state_dict(torch.load(full_path, map_location=device)['model_state_dict'])
    pca = PCA(n_components=pc) 
    embedding_learned = model.token_embed
    embedding_matrix = embedding_learned.weight.detach()
    X = embedding_matrix.cpu().numpy() 
    pca.fit(X)
    X_transformed = pca.transform(X) 
    
    x = X_transformed[:, 0]
    y = X_transformed[:, 1]
    z = X_transformed[:, 2]

    fig = plt.figure(figsize=(6, 18))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, alpha=0.5, s=10)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('Token Embeddings in PCA Space')

    plt.tight_layout()
    plt.show()