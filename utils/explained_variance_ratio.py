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

    try:
        pca.fit(X)
    except Exception:
        pca = PCA(n_components=None)
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



''' Returns a plot for accuracy just like x '''
def accuracy_overdt(test, train):
    epochs = [x['epoch'] for x in test]
    test_acc = [x['test_accuracy'] for x in test]

    plt.figure(figsize=(10, 6))

    plt.plot(epochs, train, label='Training accuracy (%)', color='#2563EB', linewidth=2)
    plt.plot(epochs, test_acc, label='Evaluation accuracy (%)', color='#059669', linewidth=2)

    plt.title('Training vs Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def overfitting_plot(train_history, eval_losses):
    # iterating two times large run itme complixity but fine for now.
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


def accuracy_plot(accuracy):

    labels = ['Accuracy', 'Error']
    sizes = [accuracy, 100 - accuracy]

    colors = ["#207023", "#FC8C84"]  
    explode = (0.1, 0) 


    plt.pie(sizes, 
            explode=explode, 
            labels=labels, 
            colors=colors,
            autopct='%1.1f%%', 
            shadow=True, 
            startangle=140)

    plt.legend(labels, title="Performance Metrics", loc="lower right")

    plt.axis('equal') 
    plt.title('Model Performance')
    plt.show()

def heat_map_raw_lattice(logit_lattice, vocab_size):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for c in range(vocab_size):
        ax = axes[c // 5][c % 5]
        ax.imshow(logit_lattice[:, :, c].cpu().numpy(), cmap='viridis')
        ax.set_title(f'c = {c}')
        ax.set_xlabel('b')
        ax.set_ylabel('a')

    plt.suptitle('logit_lattice[:, :, c] for all c')
    plt.tight_layout()
    plt.show()


''' For multiple plots d-mass, loss, accuracy '''
def plot_checkpoints(checkpoint_files, checkpoint_dir='../checkpoints/temp'):
    
    fig, axes = plt.subplots(len(checkpoint_files), 3, figsize=(18, 5 * len(checkpoint_files)))
    

    if len(checkpoint_files) == 1:
        axes = axes.reshape(1, -1)
    
    for i, filename in enumerate(checkpoint_files):
        full_path = os.path.join(checkpoint_dir, filename)
        checkpoint = torch.load(full_path)

        train_plot     = checkpoint['train_loss_history']
        eval_plot      = checkpoint['eval_loss_history']
        train_accuracy = checkpoint.get("train_accuracy")
        test_accuracy  = checkpoint.get("test_accuracy")
        d_mass_history = checkpoint.get("d_mass")
        dimension      = checkpoint.get("d_model")
        weight_decay   = checkpoint.get("weight_decay")
        vocab_size     = checkpoint.get("vocab_size")

        train_losses   = [d['loss'] for d in train_plot]
        train_epochs   = [d['epoch'] for d in train_plot]
        test_acc_vals  = [d['test_accuracy'] for d in test_accuracy]

        # loss
        axes[i, 0].plot(train_epochs, train_losses, color='red', label='Training Loss')
        axes[i, 0].plot(eval_plot, color='orange', label='Evaluation Loss')
        axes[i, 0].set_title(f'{filename} — Loss (wd={weight_decay}, mod={vocab_size})')
        axes[i, 0].set_xlabel('Epochs')
        axes[i, 0].set_ylabel('Loss')
        axes[i, 0].legend()
        axes[i, 0].grid(True, linestyle='--', alpha=0.7)

        # accuracy
        axes[i, 1].plot(train_accuracy, color='blue', label='Training Accuracy (%)')
        axes[i, 1].plot(test_acc_vals, color='green', label='Evaluation Accuracy (%)')
        axes[i, 1].set_title(f'{filename} — Accuracy')
        axes[i, 1].set_xlabel('Epochs')
        axes[i, 1].set_ylabel('Accuracy (%)')
        axes[i, 1].legend()
        axes[i, 1].grid(True, linestyle='--', alpha=0.7)

        # diagonal-mass
        if d_mass_history:
            axes[i, 2].plot(d_mass_history, color='purple', linewidth=2)
            axes[i, 2].set_title(f'{filename} — D-mass (d_model={dimension})')
            axes[i, 2].set_xlabel('Epoch')
            axes[i, 2].set_ylabel('Diagonal Spectral Mass')
            axes[i, 2].grid(True, linestyle='--', alpha=0.7)
        else:
            axes[i, 2].text(0.5, 0.5, 'No D-mass data',
                            ha='center', va='center',
                            transform=axes[i, 2].transAxes)

    plt.tight_layout()
    plt.show()
