import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)

    num_rows = len(seqs)

    if not seqs:
        return np.empty((0, max_len if max_len else 0))

    padded_seqs = np.full((num_rows, max_len), pad_value)
    for i, seq in enumerate(seqs):
        actual_len = min(len(seq), max_len) 
        padded_seqs[i, :actual_len] = seq[:actual_len]
    
    return padded_seqs
    