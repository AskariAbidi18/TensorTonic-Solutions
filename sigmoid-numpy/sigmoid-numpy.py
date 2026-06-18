import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    x = np.array(x)

    sigma_x = (1/(1 + np.exp(-x)))

    return sigma_x 
    