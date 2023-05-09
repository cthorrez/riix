"""
naive rating systems for baseline comparisons
"""

import math
import time
import numpy as np
from raters import Rater
from utils import sigmoid
from preprocessing import (index_players,
                          Virtual1v1Processor)



class CountBasedRater(Rater):
    def __init__(self, num_players, criterion='winrate', temperature=1, **kwargs):
        super().__init__(**kwargs)
        self.wins = np.zeros(num_players+1)
        self.games = np.zeros(num_players+1)
        self.correct = 0
        self.total = 0
        self.criterion = criterion
        self.num_players = num_players
        self.temperature = temperature

    def get_criterion(self, idxs):
        if self.criterion == 'wins':
            criterion = self.wins[idxs]
        elif self.criterion == 'games':
            criterion = self.games[idxs]
        elif self.criterion == 'winrate':
            criterion = self.wins[idxs] / np.clip(self.games[idxs], 1, None)
        return criterion

    def fit_predict_batch(self, batch):
        idxs = batch[self.team1_cols + self.team2_cols].values
        team1_idxs, team2_idxs = np.hsplit(idxs, 2)
        players_per_team = team1_idxs.shape[1]
        scores = batch[self.score_col].values

        active_in_period = np.unique(idxs)
        active_in_period = active_in_period[active_in_period!=-1]
        team1_mask = np.equal(team1_idxs[:,:,None], active_in_period[None,:])
        team2_mask = np.equal(team2_idxs[:,:,None], active_in_period[None,:])
        
        criterion = self.get_criterion(idxs)
        criterion[np.isnan(criterion)] = 0
        criterion1, criterion2 = np.hsplit(criterion,2)
        criterion1, criterion2 = criterion1.mean(axis=1), criterion2.mean(axis=1)

        preds = sigmoid(criterion1 - criterion2, self.temperature)

        self.games[active_in_period] += team1_mask.sum(axis=(0,1)) + team2_mask.sum(axis=(0,1))
        self.wins[active_in_period] += (team1_mask & scores[:,None,None]).sum(axis=(0,1))
        self.wins[active_in_period] += (team2_mask & (1 - scores)[:,None,None]).sum(axis=(0,1))
        return preds

    def rank(self, n=10):
        idxs = np.argsort(self.get_criterion(np.arange(self.num_players)))[::-1]
        printed = 0
        i = 0
        print(f'{"player": <26}{"win rate": <12}{"wins": <12}games')
        while printed < n:
            i += 1
            if (self.wins[idxs[i]] == 0) or (self.games[idxs[i]] < 10):
                continue
            out = f'{self.idx_to_player[idxs[i]]: <26}{round(self.wins[idxs[i]] / self.games[idxs[i]],2): <12}{int(self.wins[idxs[i]]): <12}{int(self.games[idxs[i]])}'
            print(out)
            printed += 1

