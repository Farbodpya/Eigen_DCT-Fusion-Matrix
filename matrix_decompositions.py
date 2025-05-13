import numpy as np
from scipy.fftpack import dct
from sklearn.decomposition import PCA

def ewf_decomposition(A, k):
    L, E = np.linalg.eigh(A @ A.T)
    idx = np.argsort(L)[::-1]
    E = E[:, idx]
    F = dct(np.eye(A.shape[1]), norm='ortho')
    W = E.T @ A @ F.T
    Ar = E[:, :k] @ W[:k, :] @ F
    error = np.linalg.norm(A - Ar, 'fro')
    return Ar, error

def svd_decomposition(A, k):
    U, S, Vt = np.linalg.svd(A)
    Asvd = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    error = np.linalg.norm(A - Asvd, 'fro')
    return Asvd, error

def dct_compression(A, k):
    F = dct(np.eye(A.shape[1]), norm='ortho')
    W = A @ F.T
    Aff = W[:, :k] @ F[:k, :]
    error = np.linalg.norm(A - Aff, 'fro')
    return Aff, error

def pca_compression(A, k):
    pca = PCA(n_components=k)
    A_pca = pca.fit_transform(A)
    A_reconstructed = pca.inverse_transform(A_pca)
    error = np.linalg.norm(A - A_reconstructed, 'fro')
    return A_reconstructed, error
