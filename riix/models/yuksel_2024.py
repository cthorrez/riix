"""
The rating system by Cem Yuksel, 2024
  * paper: http://www.cemyuksel.com/research/matchmaking/i3d2024-matchmaking.pdf
  * supplemental: http://www.cemyuksel.com/research/matchmaking/i3d2024-matchmaking-supplemental.pdf
  * webpage: http://www.cemyuksel.com/research/matchmaking/
"""
import math
import numpy as np
from riix.core.base import OnlineRatingSystem
from riix.utils.math_utils import sigmoid, sigmoid_scalar
from riix.utils.constants import Q, Q2_3_OVER_PI2


class Yuksel2024(OnlineRatingSystem):
    """
    The rating system presented in Skill-Based Matchmaking for Competitive Two-Player Games by Cem Yuksel
    """

    rating_dim = 2

    def __init__(
        self,
        competitors: list,
        initial_rating: float = 1500.0,
        delta_r_max: float = 350.0,
        alpha: float = 2.0,
        scaling_factor: float = 0.9,
        update_method: str = 'iterative',
        dtype=np.float64,
    ):
        """Initialize the rating system"""
        super().__init__(competitors)
        self.delta_r_max = delta_r_max
        self.alpha = alpha
        self.scaling_factor = scaling_factor
        self.ratings = np.zeros(shape=len(self.competitors), dtype=dtype) + initial_rating
        self.R = np.zeros(shape=len(self.competitors), dtype=dtype)
        self.W = np.zeros(shape=len(self.competitors), dtype=dtype) + 1e-2
        self.V = np.zeros(shape=len(self.competitors), dtype=dtype)
        self.D = np.zeros(shape=len(self.competitors), dtype=dtype)

        if update_method == 'batched':
            self.update = self.batched_update
        elif update_method == 'iterative':
            self.update = self.iterative_update

    def predict(self, matchups: np.ndarray, time_step: int = None, set_cache: bool = False):
        """
        Generates predictions for a series of matchups between competitors.
        """
        ratings = self.ratings[matchups]
        return sigmoid(Q * (ratings[:, 0] - ratings[:, 1]))

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

    @staticmethod
    def g_scalar(phi):
        return 1.0 / math.sqrt(1.0 + (Q2_3_OVER_PI2 * (phi**2.0)))

    def iterative_update(self, matchups, outcomes, **kwargs):
        """
        Treats the matchups in the rating period as sequential events.

        Parameters:
            matchups: Sequential matchups in the rating period.
            outcomes: Results of each matchup.
        """
        for (comp_1, comp_2), outcome in zip(matchups, outcomes):
            r_1 = self.ratings[comp_1]
            r_2 = self.ratings[comp_2]
            prob = sigmoid_scalar(Q * (r_1 - r_2))
            phi_1 = math.sqrt(self.V[comp_1] / self.W[comp_1])
            phi_2 = math.sqrt(self.V[comp_2] / self.W[comp_2])
            g_1 = self.g_scalar(phi_1)
            g_2 = self.g_scalar(phi_2)
            g_alpha_1 = self.g_scalar(self.alpha * phi_1)
            g_alpha_2 = self.g_scalar(self.alpha * phi_2)
            outcome_delta = outcome - prob
            F_1 = g_2 * outcome_delta
            F_2 = -g_1 * outcome_delta
            second_deriv = Q * prob * (1.0 - prob)
            D_1 = self.D[comp_1]
            D_1 = (g_2 * second_deriv) + (g_alpha_1 * D_1)
            D_2 = self.D[comp_2]
            D_2 = (g_1 * second_deriv) + (g_alpha_2 * D_2)
            delta_r = ((D_1 * F_1) - (D_2 * F_2)) / ((D_1**2.0) + (D_2**2.0))
            delta_r = min(self.delta_r_max, max(-self.delta_r_max, delta_r))
            if math.fabs(delta_r) != 0.0:
                D_1 = F_1 / delta_r
                D_2 = -F_2 / delta_r
            self.D[comp_1] = D_1
            self.D[comp_2] = D_2

            # UpdateStats 1
            scaled_update = self.scaling_factor * delta_r
            new_r_1 = r_1 + scaled_update
            self.ratings[comp_1] = new_r_1
            omega_1 = g_alpha_1
            self.W[comp_1] = omega_1 * self.W[comp_1] + 1.0
            delta_R_1 = new_r_1 - self.R[comp_1]
            self.R[comp_1] += delta_R_1 / self.W[comp_1]
            self.V[comp_1] = (omega_1 * self.V[comp_1]) + delta_R_1 * (new_r_1 - self.R[comp_1])

            # UpdateStats 2
            new_r_2 = r_2 - scaled_update
            self.ratings[comp_2] = new_r_2
            omega_2 = g_alpha_2
            self.W[comp_2] = omega_2 * self.W[comp_2] + 1.0
            delta_R_2 = new_r_2 - self.R[comp_2]
            self.R[comp_2] += delta_R_2 / self.W[comp_2]
            self.V[comp_2] = (omega_2 * self.V[comp_2]) + delta_R_2 * (new_r_2 - self.R[comp_2])

    def print_leaderboard(self, num_places):
        sorted_idxs = np.argsort(-self.ratings)[:num_places]
        max_len = min(np.max([len(comp) for comp in self.competitors] + [10]), 25)
        print(f'{"competitor": <{max_len}}\t{"rating"}')
        for p_idx in range(num_places):
            comp_idx = sorted_idxs[p_idx]
            print(f'{self.competitors[comp_idx]: <{max_len}}\t{self.ratings[comp_idx]:.6f}')
