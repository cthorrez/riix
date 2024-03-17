"""The Elo-Davidson rating system"""
import numpy as np
import math
from riix.core.base import OnlineRatingSystem
from riix.utils.math_utils import sigmoid


class EloDavidson(OnlineRatingSystem):
    """
    Implements the Elo-Davidson rating system from https://www.researchgate.net/publication/341384358_Understanding_Draws_in_Elo_Rating_Algorithm
    This method applies the method of handling draws proposed by Davidson to the "online" Elo rating system
    Davidson's paper: https://www.jstor.org/stable/2283595
    """

    rating_dim = 1

    def __init__(
        self,
        competitors: list,
        initial_rating: float = 0.0,  # this does nothing
        k: float = 32.0,
        kappa: float = 1.0,  # kappa = 2, sigma = 200 is equivalent to Elo with scale = 400.0
        base: float = 10.0,
        sigma: float = 200.0,
        update_method: str = 'iterative',
        dtype=np.float64,
    ):
        """
        Initializes the Elo rating system with the given parameters.

        Parameters:
            competitors (list): A list of competitors to be rated within the system.
            initial_rating (float, optional): The initial Elo rating for new competitors. Defaults to 1500.0.
        """
        super().__init__(competitors)
        self.k = k
        self.kappa = kappa
        self.kappa_over_2 = kappa / 2.0
        self.alpha = math.log(base) / (2.0 * sigma)
        self.ratings = np.zeros(shape=self.num_competitors, dtype=dtype) + initial_rating
        if update_method == 'batched':
            self.update = self.batched_update
        elif update_method == 'iterative':
            self.update = self.iterative_update

    def predict(self, matchups: np.ndarray, time_step: int = None, set_cache: bool = False):
        """
        Generates predictions for a series of matchups between competitors.
        Parameters:
            matchups (np.ndarray): A NumPy array of matchups, where each row represents a matchup
                                    and contains two integers indicating the indices of the competitors
                                    in the 'ratings' array.
            time_step (int, optional): A time step at which the predictions are made. This parameter
                                    is not used in the current implementation but can be utilized
                                    for time-dependent predictions. Defaults to None.
            set_cache (bool, optional): If True, caches the computed probabilities in the 'cache'
                                        attribute under the key 'probs'. Defaults to False.

        Returns:
            np.ndarray: A NumPy array containing the predicted probabilities for the first competitor
                        in each matchup winning against the second.
        """
        ratings_1 = self.ratings[matchups[:, 0]]
        ratings_2 = self.ratings[matchups[:, 1]]
        probs = sigmoid(self.alpha * (ratings_1 - ratings_2))
        return probs

    def get_pre_match_ratings(self, matchups: np.ndarray, **kwargs):
        return self.ratings[matchups]

    def batched_update(self, matchups, outcomes, use_cache):
        """
        Apply a single update based on all results of the rating period.

        Parameters:
            matchups: Matchup information for the rating period.
            outcomes: Results of the matchups.
            use_cache: Flag to use cached probabilities or calculate anew.
        """
        raise NotImplementedError

    def iterative_update(self, matchups, outcomes, **kwargs):
        """
        Treats the matchups in the rating period as sequential events.

        Parameters:
            matchups: Sequential matchups in the rating period.
            outcomes: Results of each matchup.
            **kwargs: Additional parameters (not used).
        """
        for idx in range(matchups.shape[0]):
            comp_1, comp_2 = matchups[idx]
            r_1 = self.ratings[comp_1]
            r_2 = self.ratings[comp_2]

            diff = math.exp(self.alpha * (r_1 - r_2))
            denom = diff + (1.0 / diff) + self.kappa

            p_1 = (diff + self.kappa_over_2) / denom
            p_2 = ((1.0 / diff) + self.kappa_over_2) / denom

            update_1 = self.k * (outcomes[idx] - p_1)
            update_2 = self.k * (1.0 - outcomes[idx] - p_2)

            self.ratings[comp_1] += update_1
            self.ratings[comp_2] += update_2

    def print_leaderboard(self, num_places):
        sorted_idxs = np.argsort(-self.ratings)[:num_places]
        max_len = min(np.max([len(comp) for comp in self.competitors] + [10]), 25)
        print(f'{"competitor": <{max_len}}\t{"rating"}')
        for p_idx in range(num_places):
            comp_idx = sorted_idxs[p_idx]
            print(f'{self.competitors[comp_idx]: <{max_len}}\t{self.ratings[comp_idx]:.6f}')
