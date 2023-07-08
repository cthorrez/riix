import math
import numpy as np
import pandas as pd
from tqdm import tqdm

class vSFK:
    def __init__(
        self,
        num_players: int,
        mu_0: float = 0.0,
        v_0: float = 1.0,
        beta: float = 1.,
        s: float = 400.,
        epsilon: float = 2e-3
    ):
        self.num_players = num_players
        self.mus = np.zeros(num_players, dtype=np.float64) + mu_0
        self.vs = np.zeros(num_players, dtype=np.float64) + v_0
        self.active_mask = np.zeros(num_players, dtype=np.bool_)
        self.pid_to_idx = {}
        self.beta = beta
        self.beta2 = beta ** 2.
        self.s = s
        self.s2 = s ** 2.
        self.epsilon = epsilon
        self.log10 = math.log(10.)
        self.log10_squared = math.log(10.) ** 2.
        self.probs = []

    @staticmethod
    def F_L(z):
        return 1. / (1. + (10. ** -z))

    def bradley_terry_prob_g_h(self, z, y):
        prob = vSFK.F_L(z)
        g = self.log10 * (y - prob)
        h = self.log10_squared * prob * (1. - prob)
        return prob, g, h
        
    
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

        self.mus[self.active_mask] = self.beta * self.mus[self.active_mask]
        self.vs[self.active_mask] = self.beta2 * self.vs[self.active_mask] + self.epsilon

        mu1 = self.mus[p1_idx]
        v1 = self.vs[p1_idx]

        mu2 = self.mus[p2_idx]
        v2 = self.vs[p2_idx]

        omega = v1 + v2
        z = (mu1 - mu2) / self.s

        prob, g, h = self.bradley_terry_prob_g_h(z, result)

        denom = (self.s2) + h * omega
        mu_update = (self.s * g) / denom

        mu1 = self.beta * mu1 + v1 * mu_update
        mu2 = self.beta * mu2 - v2 * mu_update

        v_update = h / denom

        v1 = v1 * (1. - v1 * v_update)
        v2 = v2 * (1. - v2 * v_update)

        self.mus[p1_idx] = mu1
        self.vs[p1_idx] = v1

        self.mus[p2_idx] = mu2
        self.vs[p2_idx] = v2

        return prob

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
            key=lambda x: x[1] - 3*x[2],
            reverse=True
        )
        for idx in range(1, k+1):
            row = sorted_players[idx-1]
            print(f'{idx:<3}{row[0]:<20}{row[1]:.6f}    {math.sqrt(row[2]):.4f}')


if __name__ == '__main__':
    from riix.datasets import get_sc2_dataset, get_melee_dataset

    df, date_col, score_col, team1_cols, team2_cols = get_sc2_dataset(path='../data/sc2_matches_5-27-2023.csv')
    # df, date_col, score_col, team1_cols, team2_cols = get_melee_dataset(tier=1)

    n = 500000
    df = df.head(n)[[*team1_cols, *team2_cols, score_col]]
    num_players = len(pd.unique(df[[*team1_cols, *team2_cols]].values.ravel('K')))
    print(f'{num_players} unique players')
    games = list(df.itertuples(index=False, name=None))

    # beta = 0.99999
    # epsilon = 1 - beta ** 2

    beta = 1.0
    epsilon = 1e-5

    model = vSFK(
        num_players=num_players,
        mu_0=0.,
        v_0=1.,
        beta=beta,
        s=1.0,
        epsilon=epsilon
    )
    probs = model.run_schedule(games)
    model.topk(25)

    acc = ((probs > 0.5) == df[score_col]).mean()
    print(f'accuracy: {acc}')
        