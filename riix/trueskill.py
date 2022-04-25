"""
An implementation of the TrueSkill rating system
https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/
https://www.moserware.com/assets/computing-your-skill/The%20Math%20Behind%20TrueSkill.pdf
"""

import math
import time
import numpy as np
from scipy.stats import norm
import pandas as pd
from raters import Rater
from preprocessing import index_players

# index for a non existant player
DUMMY = -1


def get_v_and_w(x, eps):
    diff = x - eps
    v = norm.pdf(diff) / norm.cdf(diff)

    bad_mask = np.isnan(v) | np.isinf(v)
    if bad_mask.any():
        v[bad_mask] = (-1*(diff))[bad_mask]
    w = v * (v + diff)
    return v, w


class TrueSkill(Rater):
    def __init__(self, num_players, initial_mu=25, initial_sigma=None, beta=None, tau=None, 
                 eps=None, draw_probability=0.1, decay_inactive=False, **kwargs):
        super().__init__(**kwargs)
        initial_sigma = initial_sigma or initial_mu / 3
        print('initial sigma:', initial_sigma)
        self.num_players = num_players
        self.mus = np.zeros(num_players+1, dtype=float) + initial_mu
        self.sigma2s = np.zeros(num_players+1, dtype=float) + initial_sigma**2
        self.beta = beta or initial_sigma / 2
        self.tau = tau or initial_sigma / 100
        self.draw_probability = draw_probability
        self.eps = eps
        self.decay_inactive = decay_inactive
        self.has_played_mask = np.zeros(num_players+1, dtype=bool)


    def fit_predict_batch(self, batch):
        idxs = batch[self.team1_cols + self.team2_cols].values
        n_games, players_per_game = idxs.shape
        n_players_per_team = players_per_game // 2
        scores = batch[self.score_col].values
        if self.eps is None:
            eps = norm.ppf((self.draw_probability + 1) / 2.) * math.sqrt(players_per_game) * self.beta
        else:
            eps = self.eps

        active_in_period = np.unique(idxs)
        masks = np.equal(idxs[:,:,None], active_in_period[None,:])

        if self.decay_inactive:
            self.has_played_mask[active_in_period] = True
            inactive = np.setdiff1d(np.arange(self.num_players), active_in_period)
            inactive_mask = np.zeros(self.num_players+1, dtype=bool)
            inactive_mask[inactive] = True
            increase_sigma_mask = inactive_mask & self.has_played_mask
        else:
            increase_sigma_mask = active_in_period
        self.sigma2s[increase_sigma_mask] = self.sigma2s[increase_sigma_mask] + self.tau**2

        mus = self.mus[idxs]
        sigma2s = self.sigma2s[idxs]
        c2 = players_per_game*(self.beta**2) + sigma2s.sum(axis=1)

        c = np.sqrt(c2)
        update_mask = (2*(scores - 0.5)).astype(int)
        mu_deltas = mus[:,:n_players_per_team].sum(axis=1) - mus[:,n_players_per_team:].sum(axis=1)
        mu_deltas = mu_deltas * update_mask
        mu_deltas_over_c = mu_deltas/c
        v, w = get_v_and_w(mu_deltas_over_c, eps/c)

        probs = norm.cdf(mu_deltas_over_c)
        probs[(1-scores).astype(bool)] = 1 - probs[(1-scores).astype(bool)]

        
        # [n_games, 2*n_players_per_team]
        mu_updates = ((sigma2s / c[:,None])*v[:,None])

        mu_updates[:,n_players_per_team:] = mu_updates[:,n_players_per_team:] * -1
        mu_updates = mu_updates * update_mask[:,None]
        
        sigma2_updates = (np.square(sigma2s)*w[:,None])/c2[:,None]

        games_played = masks.sum((0,1))
        mu_updates_active = (mu_updates[:,:,None] * masks).sum((0,1))
        sigma2_updates_active = (sigma2_updates[:,:,None] * masks).sum((0,1)) / games_played

        self.mus[active_in_period] += mu_updates_active
        self.sigma2s[active_in_period] -= sigma2_updates_active
        return probs

    def rank(self, n=5):
        idxs = np.argsort(self.mus[:-1] - 3*np.sqrt(self.sigma2s[:-1]))[::-1]
        print(f'{"player": <26}{"mu": <12}sigma')
        for i in range(n):
            out = f'{self.idx_to_player[idxs[i]]: <26}{round(self.mus[idxs[i]],6): <12}{round(math.sqrt(self.sigma2s[idxs[i]]),6)}'
            print(out)

if __name__ == '__main__':
    model = TrueSkill(num_players=4, eps=0)

    # data = [
    #     {'a1' : '0', 'a2' : '1', 'b1' : '2', 'b2': '3', 'score' : 1},
    #     {'a1' : '0', 'a2' : '2', 'b1' : '3', 'b2': '1', 'score' : 0},
    # ]
    # team1_cols = ['a1', 'a2']
    # team2_cols = ['b1', 'b2']
    data = [
        {'a1' : '0', 'b1' : '1', 'score' : 1},
        {'a1' : '0', 'b1' : '2', 'score' : 1},
        {'a1' : '2', 'b1' : '3', 'score' : 0},
        # {'a1' : '1', 'b1' : '2', 'score' : 0},
    ]
    team1_cols = ['a1']
    team2_cols = ['b1']

    df = pd.DataFrame(data)
    preds = model.fit_predict(df, team1_cols, team2_cols, 'score', 'minibatch', batch_size=1)

    model.rank(len(model.mus)-1)




        

    
            