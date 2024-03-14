"""The Online Rao Kupper rating system"""
import numpy as np
import math
from riix.core.base import OnlineRatingSystem
from riix.utils.math_utils import sigmoid, sigmoid_scalar


class OnlineRaoKupper(OnlineRatingSystem):
    """
    Implements an online version of the Rao Kupper model
    """

    rating_dim = 1

    def __init__(
        self,
        competitors: list,
        initial_rating: float = 1500.0,  # this does nothing
        theta: float = 2.5,
        step_size: float = 32.0,
        temperature: float = math.log(10.0) / 400.0,
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
        self.theta = theta
        self.log_theta = math.log(theta)
        self.step_size = step_size
        self.temperature = temperature
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
        probs = sigmoid(self.temperature * (ratings_1 - ratings_2))
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
            p_1 = sigmoid_scalar(self.temperature * (r_1 - r_2 - self.log_theta))
            p_2 = sigmoid_scalar(self.temperature * (r_2 - r_1 - self.log_theta))
            if outcomes[idx] == 1.0:
                g_1 = 1.0 - p_1
                g_2 = -p_2
            elif outcomes[idx] == 0.0:
                g_1 = -p_1
                g_2 = 1.0 - p_2
            else:
                g_1 = 0.5 - p_1
                g_2 = 0.5 - p_2

            # else:  # outcomes[idx] == 0.5
            #     p_1 = sigmoid_scalar(self.temperature * (r_1 - r_2 - self.log_theta))
            #     p_2 = sigmoid_scalar(self.temperature * (r_2 - r_1 - self.log_theta))
            #     prob = 1.0 - p_1 - p_2
            #     grad = 0.5 - prob

            self.ratings[comp_1] += self.step_size * g_1
            self.ratings[comp_2] += self.step_size * g_2

    def print_leaderboard(self, num_places):
        sorted_idxs = np.argsort(-self.ratings)[:num_places]
        max_len = min(np.max([len(comp) for comp in self.competitors] + [10]), 25)
        print(f'{"competitor": <{max_len}}\t{"rating"}')
        for p_idx in range(num_places):
            comp_idx = sorted_idxs[p_idx]
            print(f'{self.competitors[comp_idx]: <{max_len}}\t{self.ratings[comp_idx]:.6f}')
