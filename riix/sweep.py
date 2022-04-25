import numpy as np
import pandas as pd
import itertools
from preprocessing import get_num_players
from glicko import Glicko
from glicko2 import Glicko2
from elo import Elo
from count_based_methods import CountBasedRater
from metrics import all_metrics, is_better
from configs import rating_systems, rocket_league_params, starcraft2_params, league_of_legends_params, melee_params
from datasets import get_dota2_dataset, get_lol_dataset, get_melee_dataset


# team1_cols = ['team1.player1.username', 'team1.player2.username', 'team1.player3.username']
# team2_cols = ['team2.player1.username', 'team2.player2.username', 'team2.player3.username']
# # team1_cols += ['team1.name']
# # team2_cols += ['team2.name']
# games = pd.read_csv('../data/rl_games.csv').drop_duplicates()
# matches = pd.read_csv('../data/rl_matches.csv').drop_duplicates()
# games['timestamp'] = pd.to_datetime(games['timestamp'])
# matches['timestamp'] = pd.to_datetime(matches['timestamp'])
# games['score'] = games['team1.game_result'].astype(int)
# matches['score'] = matches['team1.match_result'].astype(int)
# matches = matches[~matches['has_substitution']]
# date_col = 'timestamp'
# matches = matches.drop(columns=['timestamp'])
# games = games.drop(columns=['score'])
# df = pd.merge(games, matches, on='match_id', how='inner')
# df = df.groupby('match_id').first().reset_index().sort_values('timestamp')
# score_col = 'score'


# team1_cols = ['player1']
# team2_cols = ['player2']
# df = pd.read_csv('../data/sc2_matches.csv', low_memory=False).drop_duplicates()
# df['player1'] = df['team1.player1.username'].astype(str) + '_' + df['team1.player1.player_id'].astype(str)
# df['player2'] = df['team2.player1.username'].astype(str) + '_' + df['team2.player1.player_id'].astype(str)
# df['timestamp'] = pd.to_datetime(df['timestamp'])
# df = df[df['team1_score'] != df['team2_score']]
# df['score'] = (df['team1_score'] > df['team2_score']).astype(int)
# # swap half of the player positions since there is significant bias
# np.random.seed(0)
# mask = np.random.rand(len(df)) > 0.5
# df.loc[mask, ['player1', 'player2']] = df.loc[mask, ['player2', 'player1']].values
# df.loc[mask, 'score']= 1 - df.loc[mask, 'score']
# date_col = 'timestamp'
# score_col = 'score'



# ProjektZero data
# team1_cols = [f'team1.player{x}' for x in range(1,6)]
# team2_cols = [f'team2.player{x}' for x in range(1,6)]
# team1_cols += ['team1']
# team2_cols += ['team2']
# df = pd.read_csv('../data/pz_lol_games.csv')
# df['date'] = pd.to_datetime(df['date'])
# date_col = 'date'
# score_col = 'score'

# df, date_col, score_col, team1_cols, team2_cols = get_lol_dataset('champion')
df, date_col, score_col, team1_cols, team2_cols = get_melee_dataset()
# df = df.query('tier <= 2')

num_players = get_num_players(df, team1_cols+team2_cols)

# metric = 'accuracy'
# metric = 'brier_score'
metric = 'log_loss'

print(num_players, 'players')
print(f'fitting on {len(df)} rows')

# sweep_params = rocket_league_params
# sweep_params = starcraft2_params
# sweep_params = league_of_legends_params
sweep_params = melee_params
rating_systems = ['glicko2']

for rating_system in rating_systems:
    best_metrics = {metric: [np.inf, -np.inf][int(is_better(1,0,metric))]}
    rating_period = sweep_params['rating_period']
    for params in itertools.product(*sweep_params[rating_system].values()):
        params_dict = {key:val for key,val in zip(sweep_params[rating_system].keys(), params)}
        if rating_system == 'count_based' : model = CountBasedRater(num_players, **params_dict)
        if rating_system == 'elo' : model = Elo(num_players, **params_dict)
        if rating_system == 'glicko' :  model = Glicko(num_players, **params_dict)
        if rating_system == 'glicko2' : model = Glicko2(num_players, **params_dict)
        preds = model.fit_predict(df, team1_cols, team2_cols, score_col, split_method='date', date_col=date_col, rating_period=rating_period)
        metrics = all_metrics(preds, df[score_col].values)
        print(params_dict)
        if is_better(metrics[metric], best_metrics[metric], metric):
            best_metrics = metrics
            best_settings = params_dict
            print(metrics)
            print(best_settings)

    print(rating_system, 'best metrics:', best_metrics)
    print(best_settings)
            
