import numpy as np

def softmax(x):
    x = np.array(x)
    if x.ndim == 1:
        x = x - np.max(x)
        exp_values = np.exp(x)
        return exp_values / np.sum(exp_values)
    elif x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        exp_values = np.exp(x)
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)
    else:
        raise ValueError("Input must be a 1D or 2D array.")