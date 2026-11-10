import numpy as np

def calculate_crack_coverage(mask):

    crack_pixels = np.count_nonzero(mask)

    total_pixels = mask.size

    percentage = (
        crack_pixels / total_pixels
    ) * 100

    return round(percentage, 2)
