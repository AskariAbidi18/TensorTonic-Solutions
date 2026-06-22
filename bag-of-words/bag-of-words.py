import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Your code here
    tokens = np.array(tokens)
    curr_index = 0
    indices = {}
    for word in vocab:
        indices[word] = curr_index
        curr_index +=1

    bow = np.zeros(curr_index, dtype = int)

    for token in tokens:
        if token in indices:
            bow[indices[token]] += 1
        
    return bow
        