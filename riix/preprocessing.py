"""
Preprocessing utilities for rating systems
"""
import sys
import re
import logging
import math
from datetime import timedelta
import numpy as np
import pandas as pd

logger = logging.getLogger()
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

DUMMY = -1

def get_num_players(df, player_cols):
    return len(pd.unique(df[player_cols].values.ravel('K')))

def index_players(df, player_cols):
    # a = pd.unique(df[player_cols].values.ravel('K'))
    # for p in a:
    #     if type(p) is not str :
    #         print(p)

    idx_to_player = sorted(pd.unique(df[player_cols].values.ravel('K').tolist()))
    player_to_idx = {player : idx for idx, player in enumerate(idx_to_player)}
    indexed_df = df.copy()
    indexed_df[player_cols] = df[player_cols].applymap(lambda x: player_to_idx[x])
    return indexed_df, idx_to_player, player_to_idx

def split_df_to_batches(df, split_method, date_col=None, rating_period=None, batch_size=None):
    assert split_method in {'date', 'minibatch'}
    if split_method == 'date':
        assert (date_col is not None) and (rating_period is not None)
        groups = df.groupby(pd.Grouper(key=date_col, freq=rating_period))
    elif split_method == 'minibatch':
        assert batch_size is not None
        groups = df.groupby(np.arange(len(df)) // batch_size)     
    logger.info(f'split into {len(groups)} batches')
    return [group for _, group in groups]


class Virtual1v1Processor:
    def __init__(self, method):
        assert method in {'aligned', 'mean', 'all_pairs'}
        self.method = method
    
    def processes_inputs(self, idxs, scores):
        team1_idxs, team2_idxs = np.hsplit(idxs,2)
        num_games , num_players_per_team = team1_idxs.shape
        cache = {'num_players_per_team' : num_players_per_team}
        
        if self.method == 'aligned':
            idxs = np.vstack([team1_idxs.ravel(), team2_idxs.ravel()]).T
            scores = np.repeat(scores, num_players_per_team)
        elif self.method == 'mean':
            mask = np.tile(1 - np.eye(2).repeat((num_players_per_team, num_players_per_team), axis=0), (num_games,1)).astype(bool)   
            idxs = np.hstack([np.tile(team1_idxs,2).ravel()[:,None], np.tile(team2_idxs,2).ravel()[:,None]])
            scores = np.repeat(scores, 2*num_players_per_team)
            cache.update({'mask' : mask, 'idxs' : idxs})
        elif self.method == 'all_pairs':
            idxs = np.vstack([np.repeat(team1_idxs.ravel(), num_players_per_team), 
                              np.tile(team2_idxs,num_players_per_team).ravel()]).T
            scores = np.repeat(scores, cache['num_players_per_team']*cache['num_players_per_team'])
        return idxs, scores, cache
    
    def process_params(self, params_list, cache):
        if self.method == 'mean':
            for params in params_list:
                means = (params.reshape(-1, order='F')
                            .reshape(-1, cache['num_players_per_team'])
                            .mean(axis=1)
                            .reshape(-1, 2, order='F')
                            .repeat(cache['num_players_per_team'], axis=0))
                params[cache['mask']] = means[cache['mask']]
            cache['idxs'][cache['mask']] = DUMMY 
            return params_list
        else:
            return params_list

    def process_preds(self, preds, cache):
        if self.method == 'aligned':
            preds = preds.reshape(-1, cache['num_players_per_team']).mean(axis=1)
        elif self.method == 'mean':
            preds = preds.reshape(-1,2*cache['num_players_per_team']).mean(axis=1)
        elif self.method == 'all_pairs':
            preds = preds.reshape(-1,cache['num_players_per_team']*cache['num_players_per_team']).mean(axis=1)
        return preds




        


def split_to_aligned_1v1s(idxs1, idxs2, scores):
    # idxs1 [num_games, num_players_per_team]
    # idxs2 [num_games, num_players_per_team]
    # scores [num_games,]
    # returns idxs [num_games * num_players_per_team, 2]
    # returns out_scores [num_games * num_players_per_team]
    idxs = np.vstack([idxs1.ravel(), idxs2.ravel()]).T
    out_scores = np.repeat(scores, idxs1.shape[1])
    return idxs, out_scores

def split_to_all_pairs_1v1s(idxs1, idxs2, scores):
    # idxs1 [num_games, num_players_per_team]
    # idxs2 [num_games, num_players_per_team]
    # scores [num_games,]
    # returns idxs [num_games * (num_players_per_team **2), 2]
    # returns out_scores [num_games * num_players_per_team]
    idxs = np.vstack([np.repeat(idxs1.ravel(), idxs1.shape[1]), np.tile(idxs2,idxs1.shape[1]).ravel()]).T
    out_scores = np.repeat(scores, math.pow(idxs1.shape[1],2))
    return idxs, out_scores

def split_to_mean_1v1s(idxs1, idxs2, scores):
    num_games , num_players_per_team = idxs1.shape
    mask = np.tile(1 - np.eye(2).repeat((num_players_per_team, num_players_per_team), axis=0), (num_games,1)).astype(bool)
    out_scores = np.repeat(scores, 2*num_players_per_team)    
    idxs = np.hstack([np.tile(idxs1,2).ravel()[:,None], np.tile(idxs2,2).ravel()[:,None]])
    return idxs, out_scores, mask

def replace_dummies_with_means(params_list, mask, num_players_per_team):
    for params in params_list:
        means = (params.reshape(-1, order='F')
                      .reshape(-1,num_players_per_team)
                      .mean(axis=1)
                      .reshape(-1, 2, order='F')
                      .repeat(num_players_per_team, axis=0))
        params[mask] = means[mask]
    return params_list
    
        
        




if __name__ == '__main__':
    idxs1 = np.array([[0,1,2],[3,4,5]])
    idxs2 = np.array([[6,7,8],[9,10,11]])
    scores = np.array([0,1])

    rs = np.arange(12+1) / 10
    RDs = np.arange(12+1) * 2

    idxs, scores, mask = split_to_mean_1v1s(idxs1, idxs2, scores)

    print(rs)
    print(idxs)
    print(mask)

    rs = rs[idxs]


    print(rs)
    print(rs.shape)
    means = rs.reshape(-1, order='F').reshape(-1,3).mean(axis=1).reshape(4,2, order='F').repeat(3, axis=0)
    # means = rs.reshape(-1,3,order='F').mean(axis=1).reshape(4,2, order='F').repeat(3, axis=0)
    print(means)

    rs[mask] = means[mask]
    print(rs)

    print(mask.shape)

