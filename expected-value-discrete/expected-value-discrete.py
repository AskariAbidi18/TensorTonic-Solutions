import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x, p = np.array(x), np.array(p)

    if not np.isclose(np.sum(p), 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError("Probabilities must sum to 1")

    return np.sum(x*p)
