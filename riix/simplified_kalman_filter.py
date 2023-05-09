"""
An implementation of "Simplified Kalman filter for online rating: one-fits-all approach"
https://arxiv.org/abs/2104.14012
"""

import math
import time
import numpy as np
from scipy.stats import norm

class SimplifiedKalmanFilter:
    def __init__(self, num_players):
        self.num_players = num_players
