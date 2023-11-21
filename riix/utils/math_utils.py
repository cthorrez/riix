"""math utility functions for rating systems"""
import math
import statistics
from scipy.special import expit


def sigmoid(x):
    """a little faster than implementing it in numpy for d < 100000"""
    return expit(x)


INV_SQRT_2 = 1.0 / math.sqrt(2.0)


def norm_cdf(x):
    """cdf of standard normal"""
    return 0.5 * (1.0 + math.erf(x * INV_SQRT_2))


STANDARD_NORMAL = statistics.NormalDist()


def norm_pdf(x):
    """pdf of standard normal"""
    return STANDARD_NORMAL.pdf(x)
