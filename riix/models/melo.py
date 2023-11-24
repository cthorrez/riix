"""Multidimensional Elo"""
import math
import numpy as np
from riix.core.base import OnlineRatingSystem
from riix.utils.math_utils import sigmoid


def generate_orthogonal_matrix(d, k):
    # Constraint check
    if d > k + 1:
        raise ValueError('d must be less than or equal to k + 1')

    # Initialize the matrix with zeros
    matrix = np.zeros((d, k))

    # Generate a base pattern
    base_pattern = np.array([1, -1] * (k // 2) + [1] * (k % 2))

    # Fill the first d-1 rows
    for i in range(d - 1):
        shift = i % k  # Calculate the shift for the pattern
        matrix[i] = np.roll(base_pattern, shift)

    # Fill the d-th row
    matrix[-1] = -np.sum(matrix[:-1], axis=0)
    return matrix


class Melo(OnlineRatingSystem):
    """Multidimensional Elo rating system, (good for rock paper scissors problems)"""

    def __init__(
        self,
        num_competitors: int,
        initial_rating: float = 1500.0,
        dimension: int = 2,  # this is the k in melo_2k, not sure why they had to use that letter when it's already used in Elo smh
        k: float = 32.0,  # this is the normal elo k factor
        alpha: float = math.log(10.0) / 400.0,
        update_method: str = 'iterative',
        dtype=np.float64,
    ):
        self.num_competitors = num_competitors
        self.k = k
        self.alpha = alpha
        self.ratings = np.zeros(shape=num_competitors, dtype=dtype) + initial_rating
        two_k = 2 * dimension
        self.c = generate_orthogonal_matrix(d=two_k, k=num_competitors).T
        self.omega = np.zeros((two_k, two_k), dtype=np.int32)
        # Set every other off-diagonal element to 1 or -1
        self.omega[np.arange(0, two_k - 1, 2), np.arange(1, two_k, 2)] = 1  # Upper off diagonal
        self.omega[np.arange(1, two_k, 2), np.arange(0, two_k - 1, 2)] = -1  # Lower off diagonal

        if update_method == 'batched':
            self.update_fn = self.batched_update
        if update_method == 'iterative':
            self.update_fn = self.iterative_update

    def predict(self, time_step: int, matchups: np.ndarray, set_cache: bool = False):
        """generate predictions"""
        ratings_1 = self.ratings[matchups[:, 0]]
        ratings_2 = self.ratings[matchups[:, 1]]
        c_1 = self.c[matchups[:, 0], :, None]  # [bs, 2k, 1]
        c_2 = self.c[matchups[:, 1], None, :]  # [bs, 1, 2k]
        elo_diff = ratings_1 - ratings_2
        tmp = np.matmul(c_2, self.omega)
        melo_diff = np.matmul(tmp, c_1)[:, 0, 0]
        probs = sigmoid((self.alpha * elo_diff) + melo_diff)
        return probs

    def fit(
        self,
        time_step: int,
        matchups: np.ndarray,
        outcomes: np.ndarray,
        use_cache: bool = False,
    ):
        self.update_fn(matchups, outcomes, use_cache=use_cache)

    def batched_update(self, matchups, outcomes, use_cache):
        """apply one update based on all of the results of the rating period"""
        active_in_period = np.unique(matchups)
        masks = np.equal(matchups[:, :, None], active_in_period[None, :])  # N x 2 x active
        probs = self.predict(time_step=None, matchups=matchups, set_cache=False)
        per_match_diff = (outcomes - probs)[:, None]
        per_match_diff = np.hstack([per_match_diff, -per_match_diff])
        per_competitor_diff = (per_match_diff[:, :, None] * masks).sum(axis=(0, 1))
        self.ratings[active_in_period] += self.k * per_competitor_diff

    def iterative_update(self, matchups, outcomes, **kwargs):
        """treat the matchups in the rating period as if they were sequential"""
        for idx in range(matchups.shape[0]):
            comp_1, comp_2 = matchups[idx]
            elo_diff = self.ratings[comp_1] - self.ratings[comp_2]
            c_1 = self.c[comp_1, None].T  # [2k,1]
            c_2 = self.c[comp_2, None].T  # [2k,1]

            melo_diff = np.dot(c_1.T, np.dot(self.omega, c_2)).flatten()
            prob = sigmoid((self.alpha * elo_diff) + melo_diff)
            delta = outcomes[idx] - prob

            rating_update = self.k * delta

            self.ratings[comp_1] += rating_update
            self.ratings[comp_2] -= rating_update

            dp_dc_1 = np.dot(self.omega, c_2).flatten()
            dp_dc_2 = np.dot(self.omega.T, c_1).flatten()

            self.c[comp_1, :] += delta * dp_dc_1
            self.c[comp_2, :] += delta * dp_dc_2
