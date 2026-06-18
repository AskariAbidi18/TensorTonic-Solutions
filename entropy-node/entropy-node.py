import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    if len(y) == 0:
        return 0.0
    y = np.array(y)
    _, freq = np.unique(y, return_counts = True)

    freq = freq/len(y)
    freq = freq[freq > 0]

    entropy = - (np.sum(freq * np.log2(freq)))

    return float(entropy)
