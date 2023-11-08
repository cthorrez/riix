"""math utility functions for rating systems"""
import numpy as np


def sigmoid(x):
    """cmon if you're reading this you already know what a sigmoid is"""
    return 1.0 / (1.0 + np.exp(-x))
