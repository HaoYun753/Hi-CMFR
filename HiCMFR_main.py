#Hi-CMFR#

### You must normalize the input matrix and rescale to 0~1 beforehand! ###

### "matrix" is your input Hi-C contact matrix, and it has to be preloaded in your environment as a 2D numpy array/matrix ###

### HYPERPARAMETER ####
## reinsertion_rate: a number between 0 and 1 ##
## n_components: a number larger than 3 ##
## conv: "gaussian" or "blurring" ##
## mf: "SVD" or "NMF" ##

def HiCMFR(matrix, reinsertion_rate=0.9, n_components=10, conv="gaussian", mf="NMF"):
    gamma=reinsertion_rate
    if n_components >= min(matrix.shape):
        n_components = min(matrix.shape) - 1
    
    blur_kernel = np.ones((3, 3)) / 9.0

    if conv=="blurring":
        matrix_conved=blur_matrix(matrix, blur_kernel)
    elif conv=="gaussian":
        matrix_conved=apply_gaussian(matrix, sigma=1)
    else:
        print("ERROR: conv must be either blurrring or gaussian.")


    if mf=="SVD":
        enhanced_matrix=process_matrix(matrix_conved,gamma, n_components)
        enhanced_matrix= np.maximum(enhanced_matrix, 0)
    elif mf=="NMF":
        enhanced_matrix=process_matrix2(matrix_conved,gamma, n_components)
        enhanced_matrix= np.maximum(enhanced_matrix, 0)
    else:
        print("ERROR: mf must be either SVD or NMF.")


    final_matrix=fill_symmetric(enhanced_matrix)
    final_matrix=normalize(final_matrix)

    return final_matrix
