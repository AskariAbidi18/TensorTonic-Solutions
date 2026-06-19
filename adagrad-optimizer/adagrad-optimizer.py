import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    # Write code here
    w, g, G = np.array(w), np.array(g), np.array(G)

    Gt = G + g**2

    wt = w - (lr / (np.sqrt(Gt+ eps))) * g

    return (wt, Gt)
    