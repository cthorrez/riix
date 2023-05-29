import math
import numpy as np
from scipy.stats import norm
import pandas as pd
from tqdm import tqdm

class Glicko:
    def __init__(
        self,
        num_players: int,
        mu_0: float = 0.0,
        v_0: float = 1.0,
        sigma = 1.0,
        epsilon: float = 2e-3
    ):
        self.num_players = num_players
        self.sigma = sigma
        self.sigma2 = sigma ** 2.
        self.a = 3. * math.pow(math.log(10.),2.) / math.pow(math.pi,2.)
        self.mus = np.zeros(num_players, dtype=np.float64) + mu_0
        self.vs = np.zeros(num_players, dtype=np.float64) + v_0
        self.active_mask = np.zeros(num_players, dtype=np.bool_)
        self.pid_to_idx = {}
        self.epsilon = epsilon
        self.log10 = math.log(10.)
        self.log10_squared = math.log(10.) ** 2.
        self.probs = []

    @staticmethod
    def F_L(z):
        return 1. / (1. + (10. ** -z))

    def bradley_terry_prob_g_h(self, z, y):
        prob = Glicko.F_L(z)
        g = self.log10 * (y - prob)
        h = self.log10_squared * prob * (1. - prob)
        return prob, g, h

    def r(self, v):
        return math.sqrt(1. + (v * self.a / self.sigma2))

        
    def update_players(self, p1_id, p2_id, result):
        # update variance of the players for passage of time

        if p1_id not in self.pid_to_idx:
            p1_idx = len(self.pid_to_idx)
            self.pid_to_idx[p1_id] = p1_idx
            self.active_mask[p1_idx] = 1
        else:
            p1_idx = self.pid_to_idx[p1_id]

        if p2_id not in self.pid_to_idx:
            p2_idx = len(self.pid_to_idx)
            self.pid_to_idx[p2_id] = p2_idx
            self.active_mask[p2_idx] = 1
        else:
            p2_idx = self.pid_to_idx[p2_id]

        self.vs[self.active_mask] += self.epsilon

        mu1 = self.mus[p1_idx]
        v1 = self.vs[p1_idx]

        mu2 = self.mus[p2_idx]
        v2 = self.vs[p2_idx]

        omega = v1 + v2

        sigma_tilde_1 = self.sigma * self.r(omega - v1)
        sigma_tilde_2 = self.sigma * self.r(omega - v2)

        delta = mu1 - mu2
        z1 = delta / sigma_tilde_1
        z2 = delta / sigma_tilde_2

        prob1, g1, h1 = self.bradley_terry_prob_g_h(z1, result)
        prob2, g2, h2 = self.bradley_terry_prob_g_h(z2, result)

        sigma2_tilde_1 = sigma_tilde_1 ** 2.
        sigma2_tilde_2 = sigma_tilde_2 ** 2.

        mu1 = mu1 + v1 * ((sigma_tilde_1 * g1) / (sigma2_tilde_1 + v1 * h1))
        mu2 = mu2 - v2 * ((sigma_tilde_2 * g2) / (sigma2_tilde_2 + v2 * h2))

        v1 = v1 * (sigma2_tilde_1 / (sigma2_tilde_1 + v1 * h1))
        v2 = v2 * (sigma2_tilde_2 / (sigma2_tilde_2 + v2 * h2))

        self.mus[p1_idx] = mu1
        self.vs[p1_idx] = v1

        self.mus[p2_idx] = mu2
        self.vs[p2_idx] = v2

        return prob1

    def play_game(self, p1_id, p2_id, result):

        prob = self.update_players(p1_id, p2_id, result)
        self.probs.append(prob)


    def run_schedule(self, games):
        for (p1_id, p2_id, result) in tqdm(games):
            self.play_game(p1_id, p2_id, result)
        return np.array(self.probs)

    def topk(self, k):
        sorted_players = sorted(
            [(id, self.mus[idx], self.vs[idx]) for id, idx in self.pid_to_idx.items()],
            key=lambda x: x[1] - 2*x[2],
            reverse=True
        )
        for idx in range(1, k+1):
            row = sorted_players[idx-1]
            print(f'{idx:<3}{row[0]:<20}{row[1]:.6f}    {math.sqrt(row[2]):.4f}')


if __name__ == '__main__':
    from riix.datasets import get_sc2_dataset, get_melee_dataset

    # df, date_col, score_col, team1_cols, team2_cols = get_sc2_dataset(path='../../data/sc2_matches_5-27-2023.csv')
    df, date_col, score_col, team1_cols, team2_cols = get_melee_dataset(tier=1)

    n = 500000
    df = df.head(n)[[*team1_cols, *team2_cols, score_col]]
    num_players = len(pd.unique(df[[*team1_cols, *team2_cols]].values.ravel('K')))
    print(f'{num_players} unique players')
    games = list(df.itertuples(index=False, name=None))
    # print('loaded data')

    # beta = 0.99999
    # epsilon = 1 - beta ** 2

    beta = 1.0
    epsilon = 1e-5

    model = Glicko(
        num_players=num_players,
        mu_0=0.,
        v_0=1.0,
        sigma=1.0,
        epsilon=1e-4
    )
    probs = model.run_schedule(games)
    model.topk(50)

    acc = ((probs > 0.5) == df[score_col]).mean()
    print(f'accuracy: {acc}')
        