import numpy as np
import pandas as pd
from preprocessing import index_players, split_by_rating_period, get_num_players
from glicko import Glicko
from naive_methods import CountRater

# player_cols = ['player1', 'player2']
# df = pd.read_csv('sample_data.csv')
# df['time'] = pd.to_datetime(df['time'])

# team1_cols = ['player1']
# team2_cols = ['player2']
# df = pd.read_csv('../data/sc2_matches.csv', low_memory=False)
# df['player1'] = df['team1.player1.username'].astype(str) + '_' + df['team1.player1.player_id'].astype(str)
# df['player2'] = df['team2.player1.username'].astype(str) + '_' + df['team2.player1.player_id'].astype(str)
# df['timestamp'] = pd.to_datetime(df['timestamp'])
# # df = df[df['team1_score'] != df['team2_score']]
# df['result'] = (df['team1_score'] > df['team2_score']).astype(int)
# date_col = 'timestamp'


# team1_cols = ['team1.player1.username', 'team1.player2.username', 'team1.player3.username']
# team2_cols = ['team2.player1.username', 'team2.player2.username', 'team2.player3.username']
# team1_cols += ['team1.name']
# team2_cols += ['team2.name']
# df = pd.read_csv('../data/rl_games.csv').drop_duplicates()
# df['timestamp'] = pd.to_datetime(df['timestamp'])
# df['result'] = df['team1.game_result'].astype(int)
# date_col = 'timestamp'


team1_cols = ['team1.player1.player_id', 'team1.player2.player_id', 'team1.player3.player_id', 'team1.player4.player_id', 'team1.player5.player_id']
team2_cols = ['team2.player1.player_id', 'team2.player2.player_id', 'team2.player3.player_id', 'team2.player4.player_id', 'team2.player5.player_id']
team1_cols = ['team1.name']
team2_cols = ['team2.name']
df = pd.read_csv('../data/lol_games.csv').drop_duplicates()
for col in team1_cols + team2_cols:
    df[col] = df[col].astype(str)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['result'] = df['team1.game_result'].astype(int)
date_col = 'timestamp'


num_players = get_num_players(df, team1_cols + team2_cols)
print(f'num matches: {len(df)}')
print(f'num players: {num_players}')
model = CountRater(num_players, '1D', criterion='winrate')
model.split_fit_predict(df, date_col, team1_cols, team2_cols)
model.rank(15)