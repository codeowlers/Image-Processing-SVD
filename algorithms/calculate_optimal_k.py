import numpy as np

# The idea is that a larger rank (i.e. a matrix with more rows or columns) would have more information
# and thus require more singular values to represent accurately.

def calculate_optimal_k(singular_values_R, singular_values_G, singular_values_B, rank):
    """
    This function calculates the optimal number of singular values to retain (i.e. the optimal value of k)
    for compressing an image, based on a set of singular values for each of the 3 color channels and the rank of the original matrix.
    """
    
    # Set a heuristic value for k based on the rank of the original matrix
    heuristic_k = 0.15 * rank
    # Initialize the optimal value of k to the heuristic value
    optimal_k = heuristic_k
    
    # Define the end index for the loop (length of singular_values minus 10)
    end_idx = len(singular_values_R) - 10
    
    # Loop over the singular values from index 0 to end_idx for each color channel
    for idx in range(0, end_idx):
        # Calculate the difference between the maximum and minimum values of the current set of 10 consecutive singular values for each color channel
        sig_diff_R = np.ptp(singular_values_R[idx:idx + 10])
        sig_diff_G = np.ptp(singular_values_G[idx:idx + 10])
        sig_diff_B = np.ptp(singular_values_B[idx:idx + 10])
        # If the difference is less than a certain threshold for all color channels, assume the singular values contain little information and set optimal_k to the current index
        if sig_diff_R < 0.32 and sig_diff_G < 0.32 and sig_diff_B < 0.32:
            optimal_k = idx
            break
    # Return the rounded minimum value of heuristic_k and optimal_k to ensure that the returned value of k is always less than or equal to the heuristic value
    return round(min(heuristic_k, optimal_k))
