import numpy as np
import pandas as pd

def get_dummy_dataset():
    df = pd.read_csv('dummy_data.csv')
    df['time'] = pd.to_datetime(df['time'])
    return df, 'time', 'result', ['player1'], ['player2']

def get_dota2_dataset(mode='both'):
    df = pd.read_csv('../data/dota2_games.csv').drop_duplicates()
    df = df.fillna(0)
    for team_num in [1,2]:
        df[f'team{team_num}'] = df[f'team{team_num}.team_name'].astype(str) + '_' + df[f'team{team_num}.team_id'].astype(int).astype(str)
        for player_num in [1,2,3,4,5]:
            df[f'team{team_num}.player{player_num}'] = df[f'team{team_num}.player{player_num}.username'].astype(str) + '_' + df[f'team{team_num}.player{player_num}.player_id'].astype(int).astype(str)
    team1_cols = []
    team2_cols = []
    if mode in {'player', 'both'}:
        team1_cols += [f'team1.player{x}' for x in range(1,6)]
        team2_cols += [f'team2.player{x}' for x in range(1,6)]
    elif mode in {'team', 'both'}:
        team1_cols += ['team1']
        team2_cols += ['team2']

    date_col = 'timestamp'
    score_col = 'score'
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['score'] = df['team1_game_result'].astype(int)
    return df, date_col, score_col, team1_cols, team2_cols

def get_lol_dataset(mode='both'):
    team1_cols = []
    team2_cols = []
    if mode in {'player', 'both', 'all'}:
        team1_cols += [f'team1.player{x}.player_id' for x in range(1,6)]
        team2_cols += [f'team2.player{x}.player_id' for x in range(1,6)]
    if mode in {'team', 'both', 'all'}:
        team1_cols += ['team1.name']
        team2_cols += ['team2.name']
    if mode in {'champion', 'all'}:
        team1_cols += [f'team1.player{x}.champion' for x in range(1,6)]
        team2_cols += [f'team2.player{x}.champion' for x in range(1,6)]

    df = pd.read_csv('../data/lol_games.csv').drop_duplicates()
    for col in team1_cols + team2_cols:
        df[col] = df[col].astype(str)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['score'] = df['team1.game_result'].astype(int)
    date_col = 'timestamp'
    score_col = 'score'
    return df, date_col, score_col, team1_cols, team2_cols


def get_melee_dataset(tier=3):
    team1_cols = ['player1.player_id']
    team2_cols = ['player2.player_id']
    df = pd.read_csv('../../data/melee_matches_5-28-2023.csv')
    for col in team1_cols + team2_cols:
        df[col] = df[col].astype(str)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['result'] = df['result'].astype(int)
    df = df[df['tier'] <= tier]
    date_col = 'timestamp'
    score_col = 'result'
    return df, date_col, score_col, team1_cols, team2_cols


def get_sc2_dataset(path):
    team1_cols = ['player1']
    team2_cols = ['player2']
    df = pd.read_csv(path, low_memory=False).drop_duplicates()
    df['player1'] = df['team1.player1.username'].astype(str) + '_' + df['team1.player1.player_id'].astype(str)
    df['player2'] = df['team2.player1.username'].astype(str) + '_' + df['team2.player1.player_id'].astype(str)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[df['team1_score'] != df['team2_score']]
    df['score'] = (df['team1_score'] > df['team2_score']).astype(int)
    # swap half of the player positions since there is significant bias
    np.random.seed(0)
    mask = np.random.rand(len(df)) > 0.5
    df.loc[mask, ['player1', 'player2']] = df.loc[mask, ['player2', 'player1']].values
    df.loc[mask, 'score']= 1 - df.loc[mask, 'score']
    date_col = 'timestamp'
    return df, date_col, 'score', team1_cols, team2_cols