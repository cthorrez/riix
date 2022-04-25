"""
Base class for rating systems
"""

import math
import time
import numpy as np
from preprocessing import (index_players, 
                          split_df_to_batches, 
                          split_to_aligned_1v1s, 
                          split_to_all_pairs_1v1s,
                          split_to_mean_1v1s,
                          replace_dummies_with_means)

class Rater:
    def __init__(self, adjust_for_base_rate=False, prior_prob=0.5, prior_games=10):
        self.adjust_for_base_rate = adjust_for_base_rate
        self.positive = prior_prob * prior_games
        self.all = prior_games
        
    def setup_fit_predict(self, df, team1_cols, team2_cols, score_col, split_method, 
                          date_col=None, rating_period=None, batch_size=None):
        df, idx_to_player, player_to_idx = index_players(df, team1_cols + team2_cols)
        self.idx_to_player = idx_to_player
        self.player_to_idx = player_to_idx
        self.team1_cols = team1_cols
        self.team2_cols = team2_cols
        self.score_col = score_col
        self.split_method = split_method
        self.date_col = date_col
        self.rating_period = rating_period
        self.batch_size = batch_size
        batches = split_df_to_batches(df, split_method, date_col, rating_period, batch_size)
        return batches

    def fit_predict(self, df, team1_cols, team2_cols, score_col, split_method, 
                    date_col=None, rating_period=None, batch_size=None):
        batches = self.setup_fit_predict(df, team1_cols, team2_cols, score_col, 
                                         split_method, date_col, rating_period, batch_size)
        preds = np.empty(len(df))
        cur_idx = 0
        for batch in batches:
            preds_batch = self.fit_predict_batch(batch)
            if self.adjust_for_base_rate:
                adjustment = (self.positive / self.all) - 0.5
                preds_batch += adjustment
                preds_batch = np.clip(preds_batch, 0, 1)
            batch_size = len(batch)
            preds[cur_idx:cur_idx+batch_size] = preds_batch
            cur_idx += batch_size
            self.positive += batch[self.score_col].sum()
            self.all += batch_size
        return preds

    def fit_predict_batch(self):
        raise NotImplementedError
        

    
