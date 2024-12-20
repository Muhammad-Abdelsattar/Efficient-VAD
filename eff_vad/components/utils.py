import numpy as np


def softmax(x: np.ndarray):
    """
    Computes the softmax of the input array.
    Args:
        x (np.ndarray): The input array.

    Returns:
        np.ndarray: The softmax of the input array.
    """
    ex = np.exp(x).astype(np.float32)
    sum = np.expand_dims(np.sum(ex,axis=1),1)
    return ex/sum