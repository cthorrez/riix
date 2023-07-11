import numpy as np
import pandas as pd

class DatasetFactory:
    @classmethod
    def load_dataset(
        cls,
        data_path: str,
        game: str,
        **kwargs
    ):
        if game == 'starcraft2':
            dataset = get_sc2_dataset(data_path=data_path, **kwargs)
        elif game == 'melee':
            dataset = get_melee_dataset(data_path=data_path, **kwargs)
        elif game == 'lol':
            dataset = get_lol_dataset(data_path=data_path, mode='team', **kwargs)
        elif game == 'dota2':
            dataset = get_dota2_dataset(data_path=data_path, mode='team', **kwargs)
        elif game == 'rl':
            dataset = get_rl_dataset(data_path=data_path, mode='team', **kwargs)
        else:
            raise SystemExit(f'Got unsuported game: {game}')
        return dataset


def get_dummy_dataset():
    df = pd.read_csv('dummy_data.csv')
    df['time'] = pd.to_datetime(df['time'])
    return df, 'time', 'result', ['player1'], ['player2']


def get_rl_dataset(data_path, level='match', **kwargs):
    df = pd.read_csv(data_path)
    df = df.rename(columns={'team1.match_result': 'score'})
    df = df[['team1.team_name', 'team2.team_name', 'score', 'timestamp']]
    # drop draws
    df = df[(df['score'] == 0) | (df['score'] == 1)]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def get_dota2_dataset(data_path, mode='both', **kwargs):
    df = pd.read_csv(data_path).drop_duplicates()
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
    df['score'] = df['team1_game_result'].astype(float)
    df = df[['team1.team_name', 'team2.team_name', 'score', 'timestamp']]
    return df

def get_lol_dataset(data_path, mode='both', **kwargs):
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

    df = pd.read_csv(data_path).drop_duplicates()
    for col in team1_cols + team2_cols:
        df[col] = df[col].astype(str)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['score'] = df['team1.game_result'].astype(float)
    df = df[['team1.name', 'team2.name', 'score', 'timestamp']]

    return df


def get_melee_dataset(data_path, tier=3, **kwargs):
    team1_cols = ['player1.player_id']
    team2_cols = ['player2.player_id']
    df = pd.read_csv(data_path)
    for col in team1_cols + team2_cols:
        df[col] = df[col].astype(str)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['score'] = df['result'].astype(int)
    df = df[df['tier'] <= tier]
    df = df[['player1.player_id', 'player2.player_id', 'score', 'timestamp']]
    return df


def get_sc2_dataset(data_path, **kwargs):
    team1_cols = ['player1']
    team2_cols = ['player2']
    df = pd.read_csv(data_path, low_memory=False).drop_duplicates()
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
    df = df[['player1', 'player2', 'score', 'timestamp']]
    return df