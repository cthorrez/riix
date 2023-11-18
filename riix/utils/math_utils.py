"""math utility functions for rating systems"""
import math
import statistics
import numpy as np


def sigmoid(x):
    """cmon if you're reading this you already know what a sigmoid is"""
    return 1.0 / (1.0 + np.exp(-x))


INV_SQRT_2 = 1.0 / math.sqrt(2.0)


def norm_cdf(x):
    """cdf of standard normal"""
    return 0.5 * (1.0 + math.erf(x * INV_SQRT_2))


STANDARD_NORMAL = statistics.NormalDist()


def norm_pdf(x):
    """pdf of standard normal"""
    return STANDARD_NORMAL.pdf(x)
