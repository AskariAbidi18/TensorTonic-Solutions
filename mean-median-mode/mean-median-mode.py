import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    # Write code here
    x = np.array(x)
    elements = len(x)

    mean = np.mean(x)

    sorted_x = np.sort(x)
    
    median = np.median(x)

    values, counts = np.unique(x, return_counts=True)
    top_index = np.argmax(counts)
    mode = values[top_index]

    return (mean, median, mode)
    