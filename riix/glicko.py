"""
An implementation of the Glicko rating system
http://www.glicko.net/glicko/glicko.pdf
http://www.glicko.net/research/glicko.pdf
"""

import math
import time
import numpy as np
import pandas as pd
from raters import Rater
from preprocessing import (index_players, 
                          Virtual1v1Processor)


# constants
q = math.log(10) / 400
q2 = q ** 2
pi2 = math.pi ** 2
# index for a non existant player
DUMMY = -1

def g(RDs):
    """Applies Batch Normalization over a 2D or 3D input (a mini-batch of 1D
    Args:
        RDs: ndarray: array of "Rating Deviations"
    Outputs: 
        ndarray, transformed RDs, same shape as input
    """
    return 1 / np.sqrt(1 + (3*q2*np.square(RDs)) / pi2)

def E(r1s, r2s, gRD1s):
    return 1 / (1 + np.power(10, -gRD1s * (r1s-r2s) / 400))


class Glicko(Rater):
    def __init__(self, num_players, initial_r=1500., initial_RD=350., c=0.0, mode='1v1', **kwargs):
        super().__init__(**kwargs)
        self.num_players = num_players
        self.rs = np.zeros(num_players+1) + initial_r
        self.RDs = np.zeros(num_players+1) + initial_RD
        self.initial_RD = initial_RD
        self.c2 = c ** 2
        self.virtual_1v1_processor = None
        if mode != '1v1' : self.virtual_1v1_processor = Virtual1v1Processor(mode)


    def fit_predict_batch(self, batch):
        idxs = batch[self.team1_cols + self.team2_cols].values
        team1_idxs, team2_idxs = np.hsplit(idxs, 2)
        players_per_team = team1_idxs.shape[1]
        scores = batch[self.score_col].values
        
        if self.virtual_1v1_processor:
            idxs, scores, cache = self.virtual_1v1_processor.processes_inputs(idxs, scores)
    
        self.RDs = np.minimum(np.sqrt(np.square(self.RDs) + self.c2), self.initial_RD)
        rs = self.rs[idxs]
        RDs = self.RDs[idxs]

        if self.virtual_1v1_processor:
            rs, RDs = self.virtual_1v1_processor.process_params((rs, RDs), cache)

        gRDs = g(RDs)
        active_in_period = np.unique(idxs)
        active_in_period = active_in_period[active_in_period!=DUMMY]
        masks = np.equal(idxs[:,:,None], active_in_period[None,:])

        E1s = E(rs[:,0], rs[:,1], gRDs[:,1])
        E2s = E(rs[:,1], rs[:,0], gRDs[:,0])

        # for predictions, use variances of both players
        pred_Es = E(rs[:,0], rs[:,1], g(RDs[:,0] + RDs[:,1]))
        
        if self.virtual_1v1_processor:
            pred_Es = self.virtual_1v1_processor.process_preds(pred_Es, cache)
        
        # tmp = (E1s * (1 - E1s))[:,None] * np.square(gRDs)[:,[1,0]]
        tmp = np.stack([E1s * (1 - E1s), E2s * (1-E2s)]).T * np.square(gRDs)[:,[1,0]]
        d2 = 1 / ((tmp[:,:,None] * masks).sum(axis=(0,1)) * q2)

        scores = np.hstack([scores[:,None], 1-scores[:,None]])
        Es = np.hstack([E1s[:,None], E2s[:,None]])

        r_num = q * ((gRDs[:,[1,0]] * (scores - Es))[:,:,None] * masks).sum(axis=(0,1))
        r_denom = (1 / np.square(self.RDs[active_in_period])) + (1 / d2)
        r_prime = self.rs[active_in_period] + (r_num / r_denom)
        RD_prime = np.sqrt(1 / r_denom)

        self.rs[active_in_period] = r_prime
        self.RDs[active_in_period] = RD_prime

        return pred_Es

    def rank(self, n=5):
        idxs = np.argsort(self.rs - 3*self.RDs)[::-1]
        print(f'{"player": <26}{"rating": <12}sigma')
        for i in range(n):
            out = f'{self.idx_to_player[idxs[i]]: <26}{round(self.rs[idxs[i]],2): <12}{round(self.RDs[idxs[i]],2)}'
            print(out)

if __name__ == '__main__':
    model = Glicko(num_players=4, c=0.0)
    model.rs[:-1] = np.array([1500,1400,1550,1700])
    model.RDs[:-1] = np.array([200, 30, 100, 300])


    data = [
        {'player1' : '0', 'player2' : '1', 'score' : 1},
        {'player1' : '0', 'player2' : '2', 'score' : 0},
        {'player1' : '0', 'player2' : '3', 'score' : 0}
    ]
    df = pd.DataFrame(data)
    preds = model.fit_predict(df, ['player1'], ['player2'], 'score', 'minibatch', batch_size=3)
    print(model.rs[0])
    print(model.RDs[0])




        

    
            