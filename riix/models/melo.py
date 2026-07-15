"""Multidimensional Elo (mElo) rating system.

Carries (ratings, c): the scalar Elo ratings plus each competitor's 2k-dim
embedding row of the C matrix, whose antisymmetric pairing through omega adds a
non-transitive (rock-paper-scissors) term to the Elo score difference. No time
dynamics.

v1 quirk preserved: the online and batched updates use gradients of *opposite
sign* for the embedding term — online moves c_1 along omega @ c_2 while batched
moves it along c_2 @ omega = -(omega @ c_2).
"""

import math
import numpy as np
import jax
import jax.numpy as jnp

from riix.core.base import OnlineRatingSystem


def generate_orthogonal_matrix(d, k):
    if d > k + 1:
        raise ValueError('d must be less than or equal to k + 1')
    matrix = np.zeros((d, k))
    base_pattern = np.array([1.0, -1.0] * (k // 2) + [1.0] * (k % 2))
    for i in range(d - 1):
        matrix[i] = np.roll(base_pattern, i % k)
    matrix[-1] = -np.sum(matrix[:-1], axis=0)
    return matrix


class Melo(OnlineRatingSystem):
    """Multidimensional Elo."""

    rating_dim = 1
    _predict_state_slots = (0, 1)  # (ratings, C-matrix rows); ratings columns use only the former

    def __init__(
        self,
        competitors: list,
        initial_rating: float = 1500.0,
        dimension: int = 1,  # the k in melo_2k
        eta_r: float = 32.0,  # the normal elo k factor
        eta_c: float = 0.125,
        alpha: float = math.log(10.0) / 400.0,
        update_method: str = 'online',
        predict_mode: str = 'prequential',
        dtype=jnp.float32,
    ):
        """
        Args:
            initial_rating (float, optional): initial rating for every competitor. Defaults to 1500.0.
            dimension (int, optional): the k of mElo_2k — number of 2-d cyclic embedding pairs capturing non-transitive interactions. Defaults to 1.
            eta_r (float, optional): learning rate of the Elo-like rating component. Defaults to 32.0.
            eta_c (float, optional): learning rate of the cyclic embedding matrix. Defaults to 0.125.
            alpha (float, optional): rating-difference scale inside the sigmoid. Defaults to log(10)/400.
            update_method / predict_mode / dtype: see `OnlineRatingSystem.__init__`.
        """
        super().__init__(competitors, dtype=dtype, update_method=update_method, predict_mode=predict_mode)
        self.initial_rating = initial_rating
        self.two_k = 2 * dimension
        self.initial_c = generate_orthogonal_matrix(d=self.two_k, k=self.num_competitors).T

        omega = np.zeros((self.two_k, self.two_k))
        omega[np.arange(0, self.two_k - 1, 2), np.arange(1, self.two_k, 2)] = 1.0
        omega[np.arange(1, self.two_k, 2), np.arange(0, self.two_k - 1, 2)] = -1.0

        self.hparams = {
            'alpha': jnp.asarray(alpha, dtype=self.dtype),
            'eta_r': jnp.asarray(eta_r, dtype=self.dtype),
            'eta_c': jnp.asarray(eta_c, dtype=self.dtype),
            'omega': jnp.asarray(omega, dtype=self.dtype),
        }

    def _initial_state(self):
        return (
            jnp.full(self.num_competitors, self.initial_rating, dtype=self.dtype),
            jnp.asarray(self.initial_c, dtype=self.dtype),
        )

    @staticmethod
    def _step_core(state, x, hparams):
        ratings, cs = state
        matchup, outcome = x
        sign_flip = jnp.array([1.0, -1.0], dtype=ratings.dtype)

        r = ratings[matchup]  # (2,)
        c = cs[matchup]  # (2, 2k)
        omega = hparams['omega']

        omega_c2 = omega @ c[1]
        melo_diff = c[0] @ omega_c2
        prob = jax.nn.sigmoid(hparams['alpha'] * (r[0] - r[1] + melo_diff))
        delta = outcome - prob

        new_r = r + hparams['eta_r'] * delta * sign_flip
        omega_c1 = omega @ c[0]
        new_c = c + hparams['eta_c'] * delta * jnp.stack([omega_c2, -omega_c1])

        ratings = ratings.at[matchup].set(new_r)
        cs = cs.at[matchup].set(new_c)
        return (ratings, cs), prob, r, c

    @staticmethod
    def _step(state, x, hparams):
        new_state, prob, _, _ = Melo._step_core(state, x, hparams)
        return new_state, prob

    @staticmethod
    def _step_period_start(state, x, hparams):
        new_state, _, r, c = Melo._step_core(state, x, hparams)
        return new_state, (r, c)

    def _gathered_probs(self, gathered):
        r, c = gathered  # (n, 2) pre-match ratings, (n, 2, 2k) pre-match embeddings
        melo_diff = ((c[:, 0] @ self.hparams['omega']) * c[:, 1]).sum(axis=1)
        return jax.nn.sigmoid(self.hparams['alpha'] * (r[:, 0] - r[:, 1] + melo_diff))

    @staticmethod
    def _batched_step(state, x, hparams):
        """One padded rating period, mirroring v1 Melo.batched_update."""
        ratings, cs = state
        matchups, outcomes, mask, _ = x
        flat_idx = matchups.reshape(-1)
        omega = hparams['omega']

        r = ratings[matchups]  # (p, 2)
        c_1 = cs[matchups[:, 0]]  # (p, 2k)
        c_2 = cs[matchups[:, 1]]

        c_1_times_omega = c_1 @ omega
        melo_diff = (c_1_times_omega * c_2).sum(axis=1)
        probs = jax.nn.sigmoid(hparams['alpha'] * (r[:, 0] - r[:, 1] + melo_diff))

        delta = (outcomes - probs) * mask  # padded matches contribute nothing
        per_match_diff = jnp.stack([delta, -delta], axis=1)  # (p, 2)
        pooled = jnp.zeros(ratings.shape[0], dtype=ratings.dtype).at[flat_idx].add(per_match_diff.reshape(-1))
        ratings = ratings + hparams['eta_r'] * pooled

        dp_dc = jnp.stack([c_2 @ omega, -c_1_times_omega], axis=1)  # (p, 2, 2k)
        c_updates = hparams['eta_c'] * per_match_diff[:, :, None] * dp_dc
        cs = cs.at[flat_idx].add(c_updates.reshape(-1, cs.shape[1]))
        return (ratings, cs), probs

    @property
    def ratings(self):
        return np.asarray(self.state[0])

    @property
    def c(self):
        return np.asarray(self.state[1])

    def leaderboard_ratings(self):
        return self.ratings
