import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    # Write code here'
    predictions = np.array(predictions)
    result = []

    for sample in predictions.T:
        values, counts = np.unique(sample, return_counts = True)
        winner = values[np.argmax(counts)]
        result.append(winner)

    return result
    
    