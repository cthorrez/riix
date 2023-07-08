import math
from collections import defaultdict
from tqdm import tqdm

class Player:
    def __init__(self, mu, v):
        self.mu = mu
        self.v = v

class vSFK:
    def __init__(
        self,
        mu_0: float = 0.0,
        v_0: float = 1.0,
        beta: float = 1.,
        s: float = 400.,
        epsilon: float = 2e-3
    ):
        self.players = defaultdict(lambda : Player(mu_0, v_0))
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
        for p_id in self.players:
            old_v = self.players[p_id].v
            new_v = self.beta2 * old_v + self.epsilon 
            self.players[p_id].v = new_v
            # self.players[p_id].v += self.epsilon

        # self.players[p1_id].v += self.epsilon
        # self.players[p2_id].v += self.epsilon

        p1 = self.players[p1_id]
        p2 = self.players[p2_id]

        omega = p1.v + p2.v
        z = (self.beta * (p1.mu - p2.mu)) / self.s

        # prob = vSFK.F_L(z)
        # g = self.g_bradley_terry(z=z, y=result)
        # h = self.h_bradley_terry(z=z, y=result)

        prob, g, h = self.bradley_terry_prob_g_h(z, result)

        denom = (self.s2) + h * omega
        mu_update = (self.s * g) / denom

        p1.mu = self.beta * p1.mu + p1.v * mu_update
        p2.mu = self.beta * p2.mu - p2.v * mu_update

        v_update = h / denom

        p1.v = p1.v * (1. - p1.v * v_update)
        p2.v = p2.v * (1. - p2.v * v_update)

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
        sorted_players = sorted([(p_id, p.mu, p.v) for p_id, p in self.players.items()], key=lambda x: x[1], reverse=True)
        for idx in range(1, k+1):
            row = sorted_players[idx]
            print(f'{idx:<3}{row[0]:<20}{row[1]:.6f}    {math.sqrt(row[2]):.4f}')


if __name__ == '__main__':
    from riix.datasets import get_sc2_dataset

    df, _, _, _, _= get_sc2_dataset()

    n = 100000
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
        