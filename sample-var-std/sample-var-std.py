import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    # Write code here
    x = np.array(x)

    mean = np.mean(x)

    s_2 = (1 / (len(x) - 1)) * np.sum((x - mean) ** 2)

    sd = s_2 ** 0.5

    return (s_2, sd)
    