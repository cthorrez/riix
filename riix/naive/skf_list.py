import math
import numpy as np
from collections import defaultdict
from tqdm import tqdm


class vSFK:
    def __init__(
        self,
        mu_0: float = 0.0,
        v_0: float = 1.0,
        beta: float = 1.,
        s: float = 400.,
        epsilon: float = 2e-3
    ):
        self.mu_0 = mu_0
        self.v_0 = v_0
        self.mus = []
        self.vs = []
        self.pid_to_idx = {}
        self.beta = beta
        self.beta2 = beta ** 2.
        self.s = s
        self.s2 = s ** 2.
        self.epsilon = epsilon
        self.log10 = math.log(10.)
        self.log10_squared = math.log(10.) ** 2.

        self.correct = 0.
        self.total = 0.

    @staticmethod
    def F_L(z):
        return 1. / (1. + (10. ** -z))

    def g_bradley_terry(self, z, y):
        return self.log10 * (y - vSFK.F_L(z))

    def h_bradley_terry(self, z, y):
        return self.log10_squared * vSFK.F_L(z) * vSFK.F_L(-z)

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
            self.mus.append(self.mu_0)
            self.vs.append(self.v_0)
        else:
            p1_idx = self.pid_to_idx[p1_id]

        if p2_id not in self.pid_to_idx:
            p2_idx = len(self.pid_to_idx)
            self.pid_to_idx[p2_id] = p2_idx
            self.mus.append(self.mu_0)
            self.vs.append(self.v_0)
        else:
            p2_idx = self.pid_to_idx[p2_id]

        self.mus = (self.beta * np.array(self.mus)).tolist()
        self.vs = (self.beta2 * np.array(self.vs) + self.epsilon).tolist()

    
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
        self.correct += (prob >= 0.5) == result
        self.total += 1

    def run_schedule(self, games):
        for (p1_id, p2_id, result) in tqdm(games):
            self.play_game(p1_id, p2_id, result)
        print(f'accuracy: {self.correct / self.total}')

    def topk(self, k):
        sorted_players = sorted(
            [(id, self.mus[idx], self.vs[idx]) for id, idx in self.pid_to_idx.items()],
            key=lambda x: x[1],
            reverse=True
        )
        for idx in range(1, k+1):
            row = sorted_players[idx]
            print(f'{idx:<3}{row[0]:<20}{row[1]:.6f}    {math.sqrt(row[2]):.4f}')


if __name__ == '__main__':
    from riix.datasets import get_sc2_dataset

    df, _, _, _, _= get_sc2_dataset()

    n = 500000
    df = df.head(n)[['player1', 'player2', 'score']]
    games = list(df.itertuples(index=False, name=None))
    # print('loaded data')

    model = vSFK(
        mu_0=0.,
        v_0=1.0,
        beta=1.0,
        s=1.0,
        epsilon=1e-5
    )
    model.run_schedule(games)
    model.topk(25)
        