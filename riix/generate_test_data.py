import numpy as np
import pandas as pd

num_players = 100
num_games = 1000000
time_freq = '6H'
num_time_periods = 365 * 4 * 10
games_per_time = num_games // num_time_periods

times = pd.date_range(pd.Timestamp.now(), periods=num_time_periods, freq=time_freq)

print(times)
print(len(times))

player_skills = np.random.normal(loc=1500, scale=300, size=num_players)
print(player_skills.min(), player_skills.max())

A, Z = np.array(["A","Z"]).view("int32") 
usernames = np.random.choice(np.arange(A,Z),size=num_players*10).astype(np.int32).view('U10')

game_dfs = []
for time in times:
    players = np.random.choice(num_players, size=(games_per_time,2))
    players[players[:,0] == players[:,1]] += np.array([0,1])
    players = np.mod(players, num_players)
    assert players[players[:,0] == players[:,1]].sum() == 0
    performances = np.random.normal(loc=player_skills[players], scale=250, size=(games_per_time,2))
    results = (performances[:,0] > performances[:,1]).astype(np.int32)

    current_usernames = usernames[players]
    
    tmp_df = pd.DataFrame(data=current_usernames, columns=['player1', 'player2'])
    tmp_df['result'] = results
    tmp_df['time'] = time
    game_dfs.append(tmp_df)

df = pd.concat(game_dfs)[['time', 'player1', 'player2', 'result']]
df.to_csv('sample_data.csv', index=False)
    
