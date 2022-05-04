from dataclasses import replace
import numpy as np
from numpy.random import default_rng
import pandas as pd

rng = default_rng(0)
num_players = 5000
num_games = 100000
time_freq = '12H'
num_time_periods = 365 * 10
games_per_time = num_games // num_time_periods
leftover = num_games - (num_time_periods * games_per_time)

times = pd.date_range(pd.Timestamp.now(), periods=num_time_periods, freq=time_freq)

print(times)
print(len(times))

player_skills = np.random.normal(loc=1500, scale=300, size=num_players)
print(player_skills.min(), player_skills.max())

A, Z = np.array(["A","Z"]).view("int32") 
usernames = np.random.choice(np.arange(A,Z),size=num_players*10).astype(np.int32).view('U10')

game_dfs = []
for idx, time in enumerate(times):
    num_games_batch = games_per_time + (leftover * (idx==0))

    player1s = rng.choice(np.arange(num_players), size=num_games_batch, replace=True)
    offsets = rng.choice(np.arange(num_players-1)+1, size=num_games_batch, replace=True)
    player2s = np.mod(player1s + offsets, num_players)
    players = np.column_stack((player1s, player2s))
    assert players[players[:,0] == players[:,1]].sum() == 0
    performances = rng.normal(loc=player_skills[players], scale=300, size=(num_games_batch,2))
    results = (performances[:,0] > performances[:,1]).astype(np.int32)

    current_usernames = usernames[players]
    

    tmp_df = pd.DataFrame(data=current_usernames, columns=['player1', 'player2'])
    tmp_df['result'] = results
    tmp_df['time'] = time
    game_dfs.append(tmp_df)

df = pd.concat(game_dfs)[['time', 'player1', 'player2', 'result']]
df.to_csv('dummy_data.csv', index=False)
    
