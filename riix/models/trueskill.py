"""TrueSkill"""
import numpy as np
from riix.core.base import OnlineRatingSystem


class TrueSkill(OnlineRatingSystem):
    """the og TrueSkill rating system shoutout to Microsoft"""

    def __init__(
        self,
        num_competitors: int,
        initial_rating: float = 1500.0,
        initial_rating_dev: float = 350.0,
        c: float = 63.2,
        dtype=np.float64,
        do_weird_prob=False,
    ):
        self.num_competitors = num_competitors
        self.initial_rating_dev = initial_rating_dev
        self.c2 = c**2.0
        self.ratings = np.zeros(shape=num_competitors, dtype=dtype) + initial_rating
        self.rating_devs = np.zeros(shape=num_competitors, dtype=dtype) + initial_rating_dev
        self.has_played = np.zeros(shape=num_competitors, dtype=np.bool_)
        self.do_weird_prob = do_weird_prob

    def predict(self, time_step: int, matchups: np.ndarray, set_cache: bool = False):
        """generate predictions"""
        ratings_1 = self.ratings[matchups[:, 0]]
        ratings_2 = self.ratings[matchups[:, 1]]
        rating_diffs = ratings_1 - ratings_2
        print(rating_diffs)

    def fit(
        self,
        time_step: int,
        matchups: np.ndarray,
        outcomes: np.ndarray,
        use_cache: bool = False,
        update_method: str = 'batched',
    ):
        if update_method == 'batched':
            self.batched_update(matchups, outcomes, use_cache)
        elif update_method == 'iterative':
            self.iterative_update(matchups, outcomes)

    def increase_rating_dev(self, matchups):
        """called once per period to model the increase in variance over time"""
        active_in_period = np.unique(matchups)
        self.has_played[active_in_period] = True
        self.rating_devs[self.has_played] = np.minimum(
            np.sqrt(np.square(self.rating_devs[self.has_played]) + self.c2), self.initial_rating_dev
        )
        return active_in_period

    def batched_update(self, matchups, outcomes, use_cache=False):
        """apply one update based on all of the results of the rating period"""
        active_in_period = self.increase_rating_dev(matchups)
        masks = np.equal(matchups[:, :, None], active_in_period[None, :])  # N x 2 x active
        print(masks)

        ratings = self.ratings[matchups]
        rating_diffs = ratings[:, 0] - ratings[:, 1]
        print(rating_diffs)

    def iterative_update(self, matchups, outcomes):
        """treat the matchups in the rating period as if they were sequential"""
        self.increase_rating_dev(matchups)
        for idx in range(matchups.shape[0]):
            comp_1, comp_2 = matchups[idx]
            rating_diff = self.ratings[comp_1] - self.ratings[comp_2]
            print(rating_diff)
