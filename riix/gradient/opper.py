import math
import numpy as np
import pandas as pd
from tqdm import tqdm


def sigmoid(x):
    return 1. / (1 + math.exp(-x))


class Opperate:
    def __init__(
        self,
        num_players: int,
        theta_0: float = 0.0,
        sigma2_0: float = 1.0,
    ):
        self.num_players = num_players
        self.sigma2_0 = sigma2_0
        self.theta = np.zeros(num_players, dtype=np.float64) + theta_0
        self.C = np.zeros((num_players, ), dtype=np.float64) + sigma2_0
        self.active_mask = np.zeros(num_players, dtype=np.bool_)
        self.pid_to_idx = {}
        self.probs = []


    def update_players(self, p1_id, p2_id, result):
        # update variance of the players for passage of time

        if p1_id not in self.pid_to_idx:
            p1_idx = len(self.pid_to_idx)
            self.pid_to_idx[p1_id] = p1_idx
        else:
            p1_idx = self.pid_to_idx[p1_id]

        if p2_id not in self.pid_to_idx:
            p2_idx = len(self.pid_to_idx)
            self.pid_to_idx[p2_id] = p2_idx
        else:
            p2_idx = self.pid_to_idx[p2_id]


        idxs = [p1_idx, p2_idx]
        thetas = self.theta[idxs]
        Cs = self.C[idxs]


        my_c = math.sqrt(2*self.sigma2_0 + Cs[0] + Cs[1])
        my_prob = sigmoid((thetas[0] - thetas[1]) / my_c)
        if result == 0:
            my_grad = -my_prob / my_c
        if result == 1:
            my_grad = (1 - my_prob) / my_c

        my_diag_hess = - my_prob * (1 - my_prob) / (my_c ** 2)

        self.theta[p1_idx] += Cs[0] * my_grad
        self.theta[p2_idx] -= Cs[1] * my_grad

        self.C[idxs] += np.square(Cs) * my_diag_hess


        return my_prob

    def play_game(self, p1_id, p2_id, result):
        prob = self.update_players(p1_id, p2_id, result)
        self.probs.append(prob)


    def run_schedule(self, games):
        for (p1_id, p2_id, result) in tqdm(games):
            self.play_game(p1_id, p2_id, result)
        return np.array(self.probs)
    
    def topk(self, k):
        sorted_players = sorted(
            [(id, self.theta[idx], self.C[idx]) for id, idx in self.pid_to_idx.items()],
            key=lambda x: x[1] - 0*x[2],
            reverse=True
        )
        for idx in range(1, k+1):
            row = sorted_players[idx-1]
            print(f'{idx:<3}{row[0]:<20}{row[1]/row[2]*math.sqrt(self.sigma2_0):.6f}    {math.sqrt(row[2]):.4f}')


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

    model = Opperate(
        num_players=num_players,
        theta_0=0.,
        sigma2_0=1e-7,
    )
    probs = model.run_schedule(games)
    model.topk(10)
    preds = probs >= 0.5
    acc = (preds == df[score_col]).mean()

    model = Opperate(
        num_players=num_players,
        theta_0=0.,
        sigma2_0=1e7,
    )
    probs = model.run_schedule(games)
    preds2 = probs >= 0.5

    print(np.alltrue(preds == preds2))

    model.topk(10)
    print(f'accuracy: {acc}')
    print(f'score mean: {df[score_col].mean()}')
        