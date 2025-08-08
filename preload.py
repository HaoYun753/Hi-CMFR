#前置函數#
import numpy as np
from scipy.sparse.linalg import svds
from scipy.ndimage import convolve
from sklearn.decomposition import NMF
from scipy.ndimage import gaussian_filter


def randomly_sample_nonzero(matrix, gamma):
    non_zero_non_na_indices = np.argwhere((matrix != 0) & (~np.isnan(matrix)))
    num_samples = int(np.ceil(len(non_zero_non_na_indices) * gamma))
    
    sampled_indices = non_zero_non_na_indices[np.random.choice(len(non_zero_non_na_indices), num_samples, replace=False)]
    sampled_values = matrix[tuple(sampled_indices.T)]
    
    sampled_matrix = matrix.copy()
    sampled_matrix[tuple(sampled_indices.T)] = 0
    
    return {'sampled_matrix': sampled_matrix, 'sampled_indices': sampled_indices, 'sampled_values': sampled_values}

def sparse_matrix_factorization(matrix, n_components):
    matrix[np.isnan(matrix)] = 0
    
    # Ensure n_components is an integer
    n_components = int(n_components)
    
    # Ensure n_components is within the valid range
    if n_components <= 0 or n_components >= min(matrix.shape):
        raise ValueError("n_components must be an integer greater than 0 and less than the smallest dimension of the matrix.")
    
    U, s, Vt = svds(matrix, k=n_components)
    
    Sigma = np.diag(s)
    reconstructed_matrix = np.dot(U, np.dot(Sigma, Vt))
    return reconstructed_matrix

def insert_sampled_values(reconstructed_matrix, sampled_indices, sampled_values):
    reconstructed_matrix[tuple(sampled_indices.T)] = sampled_values
    return reconstructed_matrix

def downsample_matrix(matrix, proportion):
    #np.random.seed(1111)  # For reproducibility
    non_zero_indices = np.argwhere(matrix != 0)
    sample_size = int(np.ceil(proportion * len(non_zero_indices)))
    sampled_indices = non_zero_indices[np.random.choice(len(non_zero_indices), sample_size, replace=False)]
    
    downsampled_matrix = matrix.copy()
    downsampled_matrix[tuple(sampled_indices.T)] = 0
    return downsampled_matrix


def process_matrix(matrix, gamma, n_components):
    
    sampling_result = randomly_sample_nonzero(matrix, gamma)
    sampled_matrix = sampling_result['sampled_matrix']
    sampled_indices = sampling_result['sampled_indices']
    sampled_values = sampling_result['sampled_values']
    
    
    reconstructed_matrix = sparse_matrix_factorization(sampled_matrix, n_components)
    
    
    final_matrix = insert_sampled_values(reconstructed_matrix, sampled_indices, sampled_values)
    
    return final_matrix

def process_matrix2(matrix, gamma, n_components):
    sampling_result = randomly_sample_nonzero(matrix, gamma)
    sampled_matrix = sampling_result['sampled_matrix']
    sampled_indices = sampling_result['sampled_indices']
    sampled_values = sampling_result['sampled_values']
    
    nmf = NMF(n_components=n_components, random_state=42)
    
    W = nmf.fit_transform(matrix)  # W is the basis matrix
    H = nmf.components_        # H is the coefficient matrix
    # Reconstructed matrix T_approx
    reconstructed_matrix = np.dot(W, H)
    # Insert sampled values back into the reconstructed matrix
    final_matrix = insert_sampled_values(reconstructed_matrix, sampled_indices, sampled_values)
    return final_matrix


def apply_gaussian(matrix, sigma=1):
    smoothed_matrix = gaussian_filter(matrix, sigma=sigma)
    return smoothed_matrix
def blur_matrix(matrix, kernel):
    return convolve(matrix, kernel, mode='constant', cval=0.0)


def fill_symmetric(matrix):
    n = matrix.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if matrix[i, j] == 0 and matrix[j, i] != 0:
                matrix[i, j] = matrix[j, i]
            elif matrix[j, i] == 0 and matrix[i, j] != 0:
                matrix[j, i] = matrix[i, j]
            elif matrix[i, j] != 0 and matrix[j, i] != 0:
                avg_value = (matrix[i, j] + matrix[j, i]) / 2
                matrix[i, j] = avg_value
                matrix[j, i] = avg_value
                
    return matrix

def normalize(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    # Avoid division by zero
    if max_val - min_val == 0:
        return np.zeros(matrix.shape)
    return (matrix - min_val) / (max_val - min_val)




