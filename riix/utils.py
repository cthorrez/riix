import numpy as np

def sigmoid(x, k=1):
    return 1 / (1 + np.exp(-x*k))