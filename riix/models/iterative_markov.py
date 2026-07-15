"""online Markov Method
https://www.degruyter.com/document/doi/10.1515/jqas-2019-0070"""

import numpy as np
import jax.numpy as jnp

from riix.core.base import OnlineRatingSystem


class onlineMarkov(OnlineRatingSystem):
    """Online Markov method."""

    rating_dim = 1

    def __init__(
        self,
        competitors: list,
        initial_rating: float = 1.0,
        c: float = 0.1,
        weight_with_prob: bool = False,  # if True this becomes "Linear Elo"
        update_method: str = 'online',
        predict_mode: str = 'prequential',
        dtype=jnp.float32,
    ):
        """
        Args:
            initial_rating (float, optional): initial rating for every competitor. Defaults to 1.0.
            c (float, optional): step size of the Markov rating transfer. Defaults to 0.1.
            weight_with_prob (bool, optional): weight the transfer by the predicted win probability ("Linear Elo"). Defaults to False.
            update_method / predict_mode / dtype: see `OnlineRatingSystem.__init__`.
        """
        super().__init__(competitors, dtype=dtype, update_method=update_method, predict_mode=predict_mode)
        self.initial_rating = initial_rating
        self.hparams = {
            'c': jnp.asarray(c, dtype=self.dtype),
            # traced 0/1 flag so toggling it does not recompile the scan
            'weight_with_prob': jnp.asarray(1.0 if weight_with_prob else 0.0, dtype=self.dtype),
        }

    def _initial_state(self):
        return (jnp.full(self.num_competitors, self.initial_rating, dtype=self.dtype),)

    @staticmethod
    def _step_core(state, x, hparams):
        (ratings,) = state
        matchup, outcome = x
        r = ratings[matchup]
        r_sum = r[0] + r[1]
        prob = r[0] / r_sum
        update = hparams['c'] * r_sum * (outcome - prob)
        update = update * jnp.where(hparams['weight_with_prob'] > 0.0, prob, 1.0)
        ratings = ratings.at[matchup].add(jnp.stack([update, -update]))
        return (ratings,), prob, r

    @staticmethod
    def _step(state, x, hparams):
        new_state, prob, _ = onlineMarkov._step_core(state, x, hparams)
        return new_state, prob

    @staticmethod
    def _step_period_start(state, x, hparams):
        new_state, _, r = onlineMarkov._step_core(state, x, hparams)
        return new_state, r

    def _gathered_probs(self, gathered):
        (r,) = gathered  # (n, 2) pre-match ratings
        return r[:, 0] / (r[:, 0] + r[:, 1])

    @property
    def ratings(self):
        return np.asarray(self.state[0])

    def leaderboard_ratings(self):
        return self.ratings
