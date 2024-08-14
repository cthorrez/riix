"""Baseline rating systems"""
import numpy as np
from typing import Literal
from riix.core.base import OnlineRatingSystem
from riix.utils.math_utils import sigmoid


class BaselineRatingSystem(OnlineRatingSystem):
    """put a docstring"""

    rating_dim = 1

    def __init__(
        self,
        competitors: list,
        mode: Literal['win_rate', 'wins', 'appearances', 'random'],
        temperature: float = 1.0, # smoothing/scaling factor on the sigmoid
        prior: int = 2, # the number of virtual games to initialize competitors with, with 50% win rate to avoid divide by zero errors
        seed: int = 0
    ):
        """Initialize the rating system"""
        super().__init__(competitors)
        self.mode = mode
        self.temperature = temperature
        self.rng = np.random.default_rng(seed=seed)
        self.wins = np.zeros(shape=self.num_competitors) + (prior // 2)
        self.appearances = np.zeros(shape=self.num_competitors) + prior

    def predict(self, matchups: np.ndarray, time_step: int = None, set_cache: bool = False):
        """
        Generates predictions for a series of matchups between competitors.
        Assumes a reparameterized Bradley-Terry model of p(i > j) = sigmoid(temperature * (strength_i - strength_j))
        The strengths are either based on count statistics such as wins, appearances, or win-rate, or randomly

        """
        if self.mode == 'win_rate':
            r_1 = self.wins[matchups[:,0]] / self.appearances[matchups[:,0]]
            r_2 = self.wins[matchups[:,1]] / self.appearances[matchups[:,1]]
        elif self.mode == 'wins':
            r_1 = self.wins[matchups[:,0]]
            r_2 = self.wins[matchups[:,1]]
        elif self.mode == 'appearances':
            r_1 = self.appearances[matchups[:,0]]
            r_2 = self.appearances[matchups[:,1]]
        else: # mode == 'random'
            r_1 = self.rng.uniform(size=matchups.shape[0])
            r_2 = self.rng.uniform(size=matchups.shape[0])
        return sigmoid(self.temperature * (r_1 - r_2))

    def get_pre_match_ratings(self, matchups: np.ndarray, **kwargs):
        raise NotImplementedError

    def update(self, matchups, outcomes, use_cache, **kwargs):
        """
        Apply a single update based on all results of the rating period, 

        Parameters:
            matchups: Matchup information for the rating period. Shape (N, 2)
            outcomes: Results of the matchups. Shape (N,)
        """
        active_in_period, active_indices = np.unique(matchups, return_inverse=True)
        active_matchups = active_indices.reshape(matchups.shape)
        
        # Update appearances
        appearance_counts = np.bincount(active_matchups.flatten(), minlength=len(active_in_period))
        self.appearances[active_in_period] += appearance_counts
        
        # Create masks for active competitors
        masks = (matchups[:, :, None] == active_in_period[None, None, :])  # Shape: (N, 2, len(active_in_period))
        
        # Calculate win counts
        win_contributions = np.stack([outcomes, 1 - outcomes], axis=1)[:, :, None]  # Shape: (N, 2, 1)
        active_win_counts = (win_contributions * masks).sum(axis=(0, 1))
        
        # Update wins
        self.wins[active_in_period] += active_win_counts
        
    def print_leaderboard(self, num_places):
        raise NotImplementedError
