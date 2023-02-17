import numpy as np
import cv2
import matplotlib.pyplot as plt

def find_num_sv(image, U, S, Vt, energy_ratio=0.95):
    """
    Plots the cumulative energy ratios for each color channel as a function of the number of singular values.
    """
    # Convert image to float32 for SVD
    img_float = np.float32(image) / 255.0

    # Compute total energy
    energy = np.sum(S ** 2)

    # Compute cumulative energy ratios
    cum_energy = np.cumsum(S ** 2) / energy

    # Find the smallest number of singular values for each channel that captures the desired energy ratio
    num_sv = np.argmax(cum_energy >= energy_ratio) + 1

    return num_sv, cum_energy