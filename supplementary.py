import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# For plotting#
def plot_enhanced_heatmaps(*matrices, titles, vmax=0.5, power=0.5):
    n = len(matrices)
    cols = 2  
    rows = 2  

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))  # Adjusted figsize for a square layout
    axes = axes.flatten()  # Flatten to make indexing easier

    for i, matrix in enumerate(matrices):
        # Apply a power transformation to emphasize smaller values
        adjusted_matrix = np.power(matrix, power)  # power < 1 enhances lower values
        
        # Plot using the white-to-red color spectrum with a max value for clipping
        sns.heatmap(adjusted_matrix, cmap='Reds', cbar=True, ax=axes[i], vmax=vmax, square=True)
        axes[i].set_title(titles[i])
        axes[i].set_xticks([])  
        axes[i].set_yticks([])  

    plt.tight_layout()
    plt.show()


# Evaluation#
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

def calculate_rmse(matrix1, matrix2):
    return np.sqrt(mean_squared_error(matrix1.flatten(), matrix2.flatten()))

def kl_divergence(P, Q):
    # Normalize to make matrices sum to 1
    P = P / np.sum(P)
    Q = Q / np.sum(Q)
    
    # Add small constant to avoid division by zero or log(0)
    P = P + 1e-09
    Q = Q + 1e-09
    
    # Compute KL divergence
    return np.sum(P * np.log(P / Q))
def calculate_ssim(A, B):
    # Ensure the matrices are of the same shape
    if A.shape != B.shape:
        raise ValueError("Input matrices must have the same dimensions.")
    
    # Calculate SSIM (requires A and B to be in the same shape and dtype)
    return ssim(A, B, data_range=A.max() - A.min())


# For AMI of TADs via Louvain#
import numpy as np
import networkx as nx
import community as community_louvain
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, jaccard_score

def extract_tads(contact_matrix, resolution=2):
    # Create a new undirected graph
    G = nx.Graph()
    num_nodes = contact_matrix.shape[0]
    G.add_nodes_from(range(num_nodes))

    # Add edges with weights to the graph
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if contact_matrix[i, j] != 0:
                G.add_edge(i, j, weight=contact_matrix[i, j])

    # Check for bad node degrees
    degrees = [degree for _, degree in G.degree()]
    avg_degree = np.mean(degrees)
    if any(deg == 0 for deg in degrees) or avg_degree < 1:  
        return []  

    # Community detection using Louvain method
    np.random.seed(442)
    partition = community_louvain.best_partition(G, resolution=resolution)
    tads = [(min([node for node in partition if partition[node] == com]), 
             max([node for node in partition if partition[node] == com])) 
            for com in set(partition.values())]
    
    return tads

def compare_tads(ref_matrix, target_matrix):
    def compare_tad_sets(tads1, tads2):
        # Convert TAD intervals to binary labels for comparison
        max_pos = max(max(tads1, key=lambda x: x[1])[1], max(tads2, key=lambda x: x[1])[1]) + 1
        labels1 = np.zeros(max_pos)
        labels2 = np.zeros(max_pos)

        for i, (start, end) in enumerate(tads1):
            labels1[start:end+1] = i + 1

        for i, (start, end) in enumerate(tads2):
            labels2[start:end+1] = i + 1

        # Calculate metrics
        ami = adjusted_mutual_info_score(labels1, labels2, average_method='arithmetic')
        return ami

    # Extract TADs from both matrices
    ref_tads = extract_tads(ref_matrix)
    target_tads = extract_tads(target_matrix)

    # Check if TAD extraction failed due to bad node degree
    if not ref_tads or not target_tads:  
        return 0

    # Compare TADs
    ami = compare_tad_sets(ref_tads, target_tads)
    return ami

