import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from matrix_decompositions import (
    ewf_decomposition,
    svd_decomposition,
    dct_compression,
    pca_compression
)

def benchmark(matrix_sizes, k_ratio=0.5):
    results = []
    times = {'EWF': [], 'SVD': [], 'DCT': [], 'PCA': []}
    errors = {'EWF': [], 'SVD': [], 'DCT': [], 'PCA': []}

    print("Running benchmarks...")
    for m, n in tqdm(matrix_sizes, desc="Benchmark Progress"):
        A = np.random.rand(m, n)
        k = int(min(m, n) * k_ratio)

        # EWF
        t1 = time.time()
        _, err = ewf_decomposition(A, k)
        t2 = time.time()
        times['EWF'].append(t2 - t1)
        errors['EWF'].append(err)

        # SVD
        t1 = time.time()
        _, err = svd_decomposition(A, k)
        t2 = time.time()
        times['SVD'].append(t2 - t1)
        errors['SVD'].append(err)

        # DCT
        t1 = time.time()
        _, err = dct_compression(A, k)
        t2 = time.time()
        times['DCT'].append(t2 - t1)
        errors['DCT'].append(err)

        # PCA
        t1 = time.time()
        _, err = pca_compression(A, k)
        t2 = time.time()
        times['PCA'].append(t2 - t1)
        errors['PCA'].append(err)

    return matrix_sizes, times, errors

def make_plots(matrix_sizes, times, errors):
    sizes = [m for m, _ in matrix_sizes]

    plt.figure(figsize=(12, 5))

    # Execution time
    plt.subplot(1, 2, 1)
    for algo in times:
        plt.plot(sizes, times[algo], label=algo)
    plt.xlabel('Matrix Size (n x n)')
    plt.ylabel('Execution Time (s)')
    plt.title('Execution Time Comparison')
    plt.legend()
    plt.grid(True)

    # Reconstruction error
    plt.subplot(1, 2, 2)
    for algo in errors:
        plt.plot(sizes, errors[algo], label=f'{algo} Error')
    plt.xlabel('Matrix Size (n x n)')
    plt.ylabel('Reconstruction Error (Frobenius Norm)')
    plt.title('Reconstruction Error Comparison')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def make_table(matrix_sizes, times, errors):
    rows = []
    for i, (m, n) in enumerate(matrix_sizes):
        for algo in times:
            rows.append([algo, f"{m}x{n}", times[algo][i], errors[algo][i]])
    df = pd.DataFrame(rows, columns=["Algorithm", "Matrix Size", "Execution Time (s)", "Reconstruction Error"])
    print(df.to_string(index=False))
    return df

if __name__ == "__main__":
    sizes = [(10, 10), (100, 100), (1000, 1000), (2000, 2000), (3000, 3000), (5000, 5000)]
    matrix_sizes, times, errors = benchmark(sizes)
    df = make_table(matrix_sizes, times, errors)
    make_plots(matrix_sizes, times, errors)
