import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def evaluate_psnr(original_image, compressed_image):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Parameters:
        original_image (ndarray): The original image.
        compressed_image (ndarray): The compressed image.

    Returns:
        float: The PSNR value between the two images.
    """
    psnr = peak_signal_noise_ratio(original_image, compressed_image,
                                   data_range=original_image.max() - original_image.min())
    return psnr


def evaluate_ssim(original_image, compressed_image):
    """
    Calculates the Structural Similarity Index (SSIM) between two images.

    Parameters:
        original_image (ndarray): The original image.
        compressed_image (ndarray): The compressed image.

    Returns:
        float: The SSIM value between the two images.
    """
    ssim = structural_similarity(original_image, compressed_image,
                                 data_range=original_image.max() - original_image.min(), multichannel=True)
    return ssim
