import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    # Write code here
    x = np.array(x)

    ReLU_x = np.maximum(0, x)

    return ReLU_x