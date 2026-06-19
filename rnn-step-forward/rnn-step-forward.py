import numpy as np

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    # Write code here
    x_t, h_prev, Wx, Wh, b = np.array(x_t), np.array(h_prev), np.array(Wx), np.array(Wh), np.array(b)

    ht = tanh((x_t @ Wx) + (h_prev @ Wh) + b)

    return ht
