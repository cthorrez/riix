import math
import numpy as np
import pandas as pd
from tqdm import tqdm

class vSFK:
    def __init__(
        self,
        num_competitors: int,
        mu_0: float = 0.0,
        v_0: float = 1.0,
        beta: float = 1.,
        # s: float = 400., # bruh
        # epsilon: float = 2e-3 # bruh
        s: float = 1.0,
        epsilon: float = 1e-4
    ):
        self.num_competitors = num_competitors
        self.mus = np.zeros(num_competitors, dtype=np.float64) + mu_0
        self.vs = np.zeros(num_competitors, dtype=np.float64) + v_0
        self.active_mask = np.zeros(num_competitors, dtype=np.bool_)
        self.pid_to_idx = {}
        self.beta = beta
        self.beta2 = beta ** 2.
        self.s = s
        self.s2 = s ** 2.
        self.epsilon = epsilon
        self.log10 = math.log(10.)
        self.log10_squared = math.log(10.) ** 2.
        self.prev_time = None

        self.probs = []

    @staticmethod
    def F_L(z):
        return 1. / (1. + (10. ** -z))

    def bradley_terry_prob_g_h(self, z, y):
        prob = vSFK.F_L(z)
        g = self.log10 * (y - prob)
        h = self.log10_squared * prob * (1. - prob)
        return prob, g, h
        
    
    def update_players(self, p1_id, p2_id, result, time_delta=0.):
        # update player activity masks
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

        # update parameters for passage of time
        beta_t = self.beta ** time_delta
        self.mus[self.active_mask] = beta_t * self.mus[self.active_mask]
        self.vs[self.active_mask] = (beta_t ** 2) * self.vs[self.active_mask] + (time_delta * self.epsilon)

        mu1 = self.mus[p1_idx]
        v1 = self.vs[p1_idx]

        mu2 = self.mus[p2_idx]
        v2 = self.vs[p2_idx]

        omega = v1 + v2
        z = (mu1 - mu2) / self.s

        prob, g, h = self.bradley_terry_prob_g_h(z, result)

        denom = (self.s2) + h * omega
        mu_update = (self.s * g) / denom

        mu1 = mu1 + v1 * mu_update
        mu2 = mu2 - v2 * mu_update

        v_update = h / denom

        v1 = v1 * (1. - v1 * v_update)
        v2 = v2 * (1. - v2 * v_update)

        self.mus[p1_idx] = mu1
        self.vs[p1_idx] = v1

        self.mus[p2_idx] = mu2
        self.vs[p2_idx] = v2

        return prob

    def play_game(self, p1_id, p2_id, result, time_delta):
        prob = self.update_players(p1_id, p2_id, result, time_delta)
        self.probs.append(prob)


    def run_schedule(self, games):
        prev_time = games[0][3]
        for row in tqdm(games):
            p1_id, p2_id, result, time = row[:4]
            time_delta = time - prev_time
            self.play_game(p1_id, p2_id, result, time_delta)
            prev_time = time
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
    df = df.head(n)[[*team1_cols, *team2_cols, score_col, date_col]]
    df['time'] = (df[date_col] - df.iloc[0][date_col]).dt.days
    df = df.drop(date_col, axis=1)
    # df['time'] = np.arange(len(df))

    num_competitors = len(pd.unique(df[[*team1_cols, *team2_cols]].values.ravel('K')))
    print(f'{num_competitors} unique players')
    games = list(df.itertuples(index=False, name=None))

    # beta = 0.998
    # epsilon = 1 - beta ** 2

    beta = 1.000001
    epsilon = 0.001

    v_0 = 1.0

    model = vSFK(
        num_competitors=num_competitors,
        mu_0=0.0,
        v_0=v_0,
        beta=beta,
        s=1.0,
        epsilon=epsilon
    )
    probs = model.run_schedule(games)
    model.topk(25)


    labels = df[score_col]

    brier_score = np.mean(np.square(probs - labels))
    log_loss = - np.mean(labels * np.log(probs) + (1-labels) * (np.log(1-probs)))

    acc = ((probs > 0.5) == labels).mean()
    print(f'accuracy: {acc}')
    print(f'log loss: {log_loss}')
    print(f'brier score: {brier_score}')
        