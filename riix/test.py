import numpy as np
import pandas as pd
from preprocessing import index_players, get_num_players
from datasets import get_dota2_dataset, get_lol_dataset, get_melee_dataset, get_sc2_dataset
from glicko import Glicko
from glicko2 import Glicko2
from elo import Elo
from count_based_methods import CountBasedRater
from trueskill import TrueSkill
from metrics import all_metrics, plot_calibration, segment_metrics, segments, year_segments
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt

# ProjektZero data
# team1_cols = [f'team1.player{x}' for x in range(1,6)]
# team2_cols = [f'team2.player{x}' for x in range(1,6)]
# team1_cols += ['team1']
# team2_cols += ['team2']
# df = pd.read_csv('../data/pz_lol_games.csv')
# df['date'] = pd.to_datetime(df['date'])
# date_col = 'date'


# team1_cols = ['team1.player1.username', 'team1.player2.username', 'team1.player3.username']
# team2_cols = ['team2.player1.username', 'team2.player2.username', 'team2.player3.username']
# # team1_cols += ['team1.name']
# # team2_cols += ['team2.name']
# games = pd.read_csv('../data/rl_games.csv').drop_duplicates()
# matches = pd.read_csv('../data/rl_matches.csv').drop_duplicates()
# games['timestamp'] = pd.to_datetime(games['timestamp'])
# matches['timestamp'] = pd.to_datetime(matches['timestamp'])
# games['score'] = games['team1.game_result'].astype(int)
# df = games

# matches['score'] = matches['team1.match_result'].astype(int)
# matches = matches[~matches['has_substitution']]
# matches = matches.drop(columns=['timestamp'])
# games = games.drop(columns=['score'])
# df = pd.merge(games, matches, on='match_id', how='inner')
# df = df.groupby('match_id').first().reset_index().sort_values('timestamp')
# date_col = 'timestamp'

# df, date_col, score_col, team1_cols, team2_cols = get_sc2_dataset()
# df = df[df.timestamp.dt.year >= 2022]
# df, date_col, score_col, team1_cols, team2_cols = get_dota2_dataset()

df, date_col, score_col, team1_cols, team2_cols = get_lol_dataset('both')
# df = df.query('(region=="North America") & (level=="Primary" | level=="Secondary")')
# df, date_col, score_col, team1_cols, team2_cols = get_melee_dataset()
# df = df.query('tier == 1 | tier==2')
# df = df.query('timestamp.dt.year>=2021')
# df = df.query('type=="offline"')

split_method = 'date'
rating_period = '1D'

split_method = 'minibatch'
batch_size = 5


num_players = get_num_players(df, team1_cols + team2_cols)
print(f'num matches: {len(df)}')
print(f'num players: {num_players}')
# Leaguepedia settings
# model = Glicko2(num_players, mode='1v1', initial_phi=323, initial_sigma=0.036, tau=0.2, eps=1e-6, adjust_for_base_rate=True)
# model = Glicko2(num_players, mode='all_pairs', initial_phi=100, initial_sigma=0.06, tau=0.2, eps=1e-6, adjust_for_base_rate=True)
# ProjektZero settings
# model = Glicko2(num_players, mode='aligned', initial_phi=210, initial_sigma=0.038, tau=0.2, eps=1e-6, adjust_for_base_rate=True)
# model = Glicko2(num_players, mode='1v1', initial_phi=350, initial_sigma=0.06, tau=0.5, eps=1e-6, adjust_for_base_rate=False)
# model = Glicko(num_players, c=3, initial_RD=350, mode='aligned', adjust_for_base_rate=False)
# model = Elo(num_players, k=24, mode='1v1', adjust_for_base_rate=False)
# model = CountBasedRater(num_players, criterion='winrate', temperature=1.0, adjust_for_base_rate=True)
model = TrueSkill(num_players, draw_probability=0, decay_inactive=False, initial_sigma=5, tau=0.01)
preds = model.fit_predict(df, team1_cols, team2_cols, score_col, split_method, date_col, rating_period, batch_size)
model.rank(15)
# exit(1)

scores = df[score_col].values
# print('base rate:', scores.mean())

df['preds'] = preds
results = segment_metrics(df, 'preds', score_col, segments)
for key in results.keys():
    print(key, 'accuracy:', results[key]['accuracy'], results[key]['num'])

# disp = CalibrationDisplay.from_predictions(scores, preds, n_bins=15, strategy='uniform')
# plt.show()

# modifier = np.roll((scores.cumsum() + 10) / (np.arange(scores.shape[0]) + 20),1) - 0.5
# modifier[0] = 0.5
# preds +=  modifier




metrics_dict = all_metrics(preds, scores)
for key, value in metrics_dict.items():
    print(f'{key}: {value}')

# df['glicko2_blue_predicted_prob'] = preds
# df[['gameid', 'glicko2_blue_predicted_prob']].to_csv('../data/glicko2_preds.csv', index=False)