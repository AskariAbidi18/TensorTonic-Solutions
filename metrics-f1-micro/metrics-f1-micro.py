import numpy as np
def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    # Write code here
    y_true, y_pred = np.array(y_true), np.array(y_pred) 
    total_samples = len(y_true)
    unique_classes = set(y_true) | set(y_pred)

    tp = {c: 0 for c in unique_classes}
    fp = {c: 0 for c in unique_classes}
    fn = {c: 0 for c in unique_classes}

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            tp[true] += 1
        else:
            fp[pred] += 1  
            fn[true] += 1 

    tn = {}
    for c in unique_classes:
        tn[c] = total_samples - (tp[c] + fp[c] + fn[c])
        global_tp = sum(tp.values())
        global_fp = sum(fp.values())
        global_fn = sum(fn.values())
        global_tn = sum(tn.values())

        accuracy = (global_tp + global_tn) / total_samples

        if (global_tp + global_fp) == 0:
            precision = 0.0
        else:
            precision = global_tp / (global_tp + global_fp)
        if (global_tp + global_fn) == 0:
            recall = 0.0
        else:
            recall = global_tp / (global_tp + global_fn)

        if (precision + recall) == 0:
            return 0.0
            
        f1_score = 2 * ((precision * recall) / (precision + recall))

        return f1_score 
        