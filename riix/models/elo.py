"""The Elo rating system"""
import math
import numpy as np
from riix.core.base import OnlineRatingSystem
from riix.utils.math_utils import sigmoid, sigmoid_scalar


class Elo(OnlineRatingSystem):
    """
    Implements the original Elo rating system.
    """

    rating_dim = 1

    def __init__(
        self,
        competitors: list,
        initial_rating: float = 1500.0,
        k: float = 32.0,
        alpha: float = math.log(10.0) / 400.0,
        update_method: str = 'iterative',
        dtype=np.float64,
    ):
        """
        Initializes the Elo rating system with the given parameters.

        Parameters:
            competitors (list): A list of competitors to be rated within the system.
            initial_rating (float, optional): The initial Elo rating for new competitors. Defaults to 1500.0.
            k (float, optional): The K-factor, which controls the rate at which ratings change. Defaults to 32.0.
            alpha (float, optional): Scaling factor used in the calculation of expected scores. Defaults to log(10) / 400.
            update_method (str, optional): Method used to update ratings ('iterative' or other methods if implemented). Defaults to 'iterative'.
            dtype: The data type for internal numpy computations. Defaults to np.float64.

        Initializes an Elo rating system with customizable settings for initial ratings, K-factor, and update method.
        """
        super().__init__(competitors)
        self.k = k
        self.alpha = alpha
        self.ratings = np.zeros(shape=self.num_competitors, dtype=dtype) + initial_rating
        self.cache = {'probs': None}
        if update_method == 'batched':
            self.update = self.batched_update
        elif update_method == 'iterative':
            self.update = self.iterative_update

    def predict(self, matchups: np.ndarray, time_step: int = None, set_cache: bool = False):
        """
        Generates predictions for a series of matchups between competitors.

        This method calculates the probability of the first competitor in each matchup winning
        based on their Elo ratings. The probabilities are computed using the sigmoid function
        applied to the rating differences, scaled by the alpha parameter.

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
        if set_cache:
            self.cache['probs'] = probs
        return probs

    def get_pre_match_ratings(self, matchups: np.ndarray, **kwargs):
        return self.ratings[matchups]

    def batched_update(self, matchups, outcomes, use_cache, **kwargs):
        """
        Apply a single update based on all results of the rating period.

        Parameters:
            matchups: Matchup information for the rating period.
            outcomes: Results of the matchups.
            use_cache: Flag to use cached probabilities or calculate anew.
        """
        active_in_period = np.unique(matchups)
        masks = np.equal(matchups[:, :, None], active_in_period[None, :])  # N x 2 x active
        if use_cache:
            probs = self.cache['probs']
        else:
            probs = self.predict(time_step=None, matchups=matchups, set_cache=False)
        per_match_diff = (outcomes - probs)[:, None]
        per_match_diff = np.hstack([per_match_diff, -per_match_diff])
        per_competitor_diff = (per_match_diff[:, :, None] * masks).sum(axis=(0, 1))
        self.ratings[active_in_period] += self.k * per_competitor_diff

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
            diff = self.ratings[comp_1] - self.ratings[comp_2]
            prob = sigmoid_scalar(self.alpha * diff)
            update = self.k * (outcomes[idx] - prob)
            self.ratings[comp_1] += update
            self.ratings[comp_2] -= update

    def print_leaderboard(self, num_places):
        sorted_idxs = np.argsort(-self.ratings)[:num_places]
        max_len = min(np.max([len(comp) for comp in self.competitors] + [10]), 25)
        print(f'{"competitor": <{max_len}}\t{"rating"}')
        for p_idx in range(num_places):
            comp_idx = sorted_idxs[p_idx]
            print(f'{self.competitors[comp_idx]: <{max_len}}\t{self.ratings[comp_idx]:.6f}')
