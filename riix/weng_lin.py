"""
An implementation of "A Bayesian Approximation Method for Online Ranking"
https://jmlr.csail.mit.edu/papers/volume12/weng11a/weng11a.pdf
"""

import math
import time
import numpy as np
from scipy.stats import norm

class WengLinRater:
    def __init__(self, num_players):
        self.num_players = num_players
