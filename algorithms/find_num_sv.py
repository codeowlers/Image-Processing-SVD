import numpy as np
import cv2
import matplotlib.pyplot as plt

def find_num_sv(S, energy_ratio=0.95):
    # Compute total energy
    energy = np.sum(S ** 2)

    # Compute cumulative energy ratios
    cum_energy = np.cumsum(S ** 2) / energy

    # Find the smallest number of singular values for each channel that captures the desired energy ratio
    num_sv = np.argmax(cum_energy >= energy_ratio) + 1

    return num_sv, cum_energy