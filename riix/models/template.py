"""a template to copy and paste when implementing a new rating system"""
import numpy as np
from riix.core.base import OnlineRatingSystem


class Template(OnlineRatingSystem):
    """put a docstring"""

    rating_dim = 1

    def __init__(
        self,
        competitors: list,
        argument: str = 'lorem ipsum',
        update_method: str = 'iterative',
        dtype=np.float64,
    ):
        """Initialize the rating system"""
        super().__init__(competitors)
        self.argument = argument
        if update_method == 'batched':
            self.update = self.batched_update
        elif update_method == 'iterative':
            self.update = self.iterative_update

    def predict(self, matchups: np.ndarray, time_step: int = None, set_cache: bool = False):
        """
        Generates predictions for a series of matchups between competitors.
        """
        return np.zeros(matchups.shape[0]) + 0.5

    def get_pre_match_ratings(self, matchups: np.ndarray, **kwargs):
        raise NotImplementedError

    def batched_update(self, matchups, outcomes, use_cache, **kwargs):
        """
        Apply a single update based on all results of the rating period.

        Parameters:
            matchups: Matchup information for the rating period.
            outcomes: Results of the matchups.
        """
        raise NotImplementedError

    def iterative_update(self, matchups, outcomes, **kwargs):
        """
        Treats the matchups in the rating period as sequential events.

        Parameters:
            matchups: Sequential matchups in the rating period.
            outcomes: Results of each matchup.
        """
        raise NotImplementedError

    def print_leaderboard(self, num_places):
        raise NotImplementedError
