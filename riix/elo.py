"""
An implementation of the Elo rating system
"""

import math
from tqdm import tqdm
import numpy as np
from raters import Rater
from preprocessing import (index_players, 
                          Virtual1v1Processor)

DUMMY = -1

def E(r1s, r2s):
    Es = 1 / (1 + np.power(10, (r2s-r1s) / 400))
    return Es


class Elo(Rater):
    def __init__(self, num_players, initial_r=1500., k=32, mode='1v1', **kwargs):
        super().__init__(**kwargs)
        self.num_players = num_players
        self.rs = np.zeros(num_players+1) + initial_r
        self.virtual_1v1_processor = None
        self.k = k
        if mode != '1v1' : self.virtual_1v1_processor = Virtual1v1Processor(mode)

    def fit_predict_batch(self, batch):
        idxs = batch[self.team1_cols + self.team2_cols].values
        team1_idxs, team2_idxs = np.hsplit(idxs, 2)
        players_per_team = team1_idxs.shape[1]
        scores = batch[self.score_col].values

        if self.virtual_1v1_processor:
            idxs, scores, cache = self.virtual_1v1_processor.processes_inputs(idxs, scores)
    
        rs = self.rs[idxs]
        if self.virtual_1v1_processor:
            rs, = self.virtual_1v1_processor.process_params((rs,), cache)

        active_in_period = np.unique(idxs)
        active_in_period = active_in_period[active_in_period!=DUMMY]
        masks = np.equal(idxs[:,:,None], active_in_period[None,:])

        Es = E(rs[:,0], rs[:,1])

        pred_Es = Es
        if self.virtual_1v1_processor:
            pred_Es = self.virtual_1v1_processor.process_preds(pred_Es, cache)
            
        scores = np.hstack([scores[:,None], 1-scores[:,None]])
        Es = np.hstack([Es[:,None], 1-Es[:,None]])

        diffs = ((scores - Es)[:,:,None] * masks).sum(axis=(0,1))
        r_prime = self.rs[active_in_period] + self.k*diffs
        self.rs[active_in_period] = r_prime
        return pred_Es
    

    def play_game(self, p1_id, p2_id, result):
        prob = self.update_players(p1_id, p2_id, result)
        self.probs.append(prob)


    def run_schedule(self, games):
        for row in tqdm(games):
            p1_id, p2_id, result = row[:3]
            self.play_game(p1_id, p2_id, result)
        return np.array(self.probs)

    def rank(self, n=5):
        idxs = np.argsort(self.rs)[::-1]
        print(f'{"player": <26}rating')
        for i in range(n):
            out = f'{self.idx_to_player[idxs[i]]: <26}{round(self.rs[idxs[i]],4)}'
            print(out)


class EloV2:
    def __init__(
        self,
        num_competitors: int,
        initial_r=1500.,
        k=32,
    ):
        self.num_competitors = num_competitors
        self.k = k
        self.rs = np.zeros(num_competitors, dtype=np.float64) + initial_r
        self.pid_to_idx = {}

        self.log10 = math.log(10.)
        self.log10_squared = math.log(10.) ** 2.
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


        r1 = self.rs[p1_idx]
        r2 = self.rs[p2_idx]

        E1 = 1. / (1. + math.exp((r2 - r1) / 400.))

        update = self.k * (result - E1)

        self.rs[p1_idx] += update
        self.rs[p2_idx] -= update

        return E1

    def play_game(self, p1_id, p2_id, result):

        prob = self.update_players(p1_id, p2_id, result)
        self.probs.append(prob)


    def run_schedule(self, games):
        for row in tqdm(games):
            p1_id, p2_id, result = row[:3]
            self.play_game(p1_id, p2_id, result)
        return np.array(self.probs)

    def topk(self, k):
        sorted_players = sorted(
            [(id, self.rs[idx]) for id, idx in self.pid_to_idx.items()],
            key=lambda x: x[1],
            reverse=True
        )
        for idx in range(1, k+1):
            row = sorted_players[idx-1]
            print(f'{idx:<3}{row[0]:<20}{row[1]:.6f}')

