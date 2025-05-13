# Matrix Decomposition Benchmark

This project implements and benchmarks various matrix decomposition techniques to assess their performance in terms of execution time and reconstruction error. The comparison includes four popular methods:

- **EWF (Eigen + DCT hybrid)**: A novel matrix decomposition method designed for image compression and optimization.
- **SVD (Singular Value Decomposition)**: A classical technique for dimensionality reduction and compression.
- **DCT (Discrete Cosine Transform)**: Used primarily for image compression and feature extraction.
- **PCA (Principal Component Analysis)**: A well-known method for reducing dimensionality in data processing.

The purpose of this benchmark is to evaluate these algorithms across different matrix sizes and quantify their performance in terms of:

- **Execution Time**: The time each method takes to process matrices of varying sizes.
- **Reconstruction Error**: The error between the original and reconstructed matrices, measured using the Frobenius norm.

## Features

- Benchmarking of **EWF**, **SVD**, **DCT**, and **PCA** on matrices with sizes ranging from 10x10 to 5000x5000.
- Execution time and reconstruction error for each method, displayed in both tabular and graphical forms.
- Clear visualizations of performance comparison using matplotlib.
- Modular design for easy integration and future enhancements.

## Installation

To get started with this project, follow these steps to clone the repository and install the required dependencies.

### Step 1: Clone the Repository

To clone the repository, open a terminal or command prompt and run the following command:

```bash
git clone https://github.com/Farbodpya/Eigen_DCT-Fusion-Matrix.git


cd matrix-benchmark

pip install -r requirements.txt


python main.py
