"""
An implementation of the Elo rating system
http://www.glicko.net/glicko/glicko.pdf
http://www.glicko.net/research/glicko.pdf
"""

import math
import time
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

    def rank(self, n=5):
        idxs = np.argsort(self.rs)[::-1]
        print(f'{"player": <26}rating')
        for i in range(n):
            out = f'{self.idx_to_player[idxs[i]]: <26}{round(self.rs[idxs[i]],4)}'
            print(out)



        

    
            