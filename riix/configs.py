import numpy as np

rating_systems = ['count_based', 'elo', 'glicko', 'glicko2']

rocket_league_params = {
    'rating_period' : '1D',
    'count_based' : {
        'criterion' : ['winrate'],
        'temperature' : np.logspace(start=-5, stop=5, num=50)
    },
    'elo' : {
        'k' : np.linspace(1, 100, num=50),
        'mode' : ['aligned', 'mean', 'all_pairs']
    },
    'glicko' : {
        'c' : [c for c in range(1,11)],
        'initial_RD' : np.linspace(50, 600, num=10),
        'mode' : ['aligned', 'mean', 'all_pairs']
    },
    'glicko2' : {
        'tau' : [0.2],
        'initial_phi' : np.linspace(10, 500, num=30),
        'initial_sigma' : np.linspace(0.001, 0.1, num=10),
        'mode' : ['aligned', 'mean', 'all_pairs']
    }  
}

starcraft2_params = {
    'rating_period' : '1D',
    'count_based' : {
        'criterion' : ['winrate'],
        'temperature' : np.logspace(start=-5, stop=5, num=50)
    },
    'elo' : {
        'k' : np.linspace(1, 100, num=50),
        'mode' : ['1v1']
    },
    'glicko' : {
        'c' : np.linspace(0.1, 12, num=25),
        'initial_RD' : np.linspace(50, 600, num=25),
        'mode' : ['1v1']
    },
    'glicko2' : {
        'tau' : [0.2],
        'initial_phi' : np.linspace(200, 600, num=20),
        'initial_sigma' : np.linspace(0.01, 0.1, num=20),
        'mode' : ['1v1']
    }  
}


league_of_legends_params = {
    'rating_period' : '1D',
    'count_based' : {
        'criterion' : ['winrate'],
        'temperature' : np.logspace(start=-5, stop=5, num=50)
    },
    'elo' : {
        'k' : np.linspace(1, 64, num=25),
        'mode' : ['aligned', 'mean', 'all_pairs']
    },
    'glicko' : {
        'c' : [c for c in range(1,11)],
        'initial_RD' : np.linspace(50, 600, num=10),
        'mode' : ['aligned', 'mean', 'all_pairs']
    },
    'glicko2' : {
        'tau' : [0.2],
        'initial_phi' : np.linspace(50, 500, num=10),
        'initial_sigma' : np.linspace(0.01, 0.1, num=10),
        'mode' : ['aligned', 'mean', 'all_pairs'],
        'adjust_for_base_rate' : [False]
    }  
}


melee_params = {
    'rating_period' : '1D',
    'count_based' : {
        'criterion' : ['winrate'],
        'temperature' : np.logspace(start=-5, stop=5, num=50)
    },
    'elo' : {
        'k' : np.linspace(1, 64, num=25),
        'mode' : ['aligned', 'mean', 'all_pairs']
    },
    'glicko' : {
        'c' : [c for c in range(1,11)],
        'initial_RD' : np.linspace(50, 600, num=10),
        'mode' : ['aligned', 'mean', 'all_pairs']
    },
    'glicko2' : {
        'tau' : [0.3,],
        'initial_phi' : np.linspace(50, 500, num=15),
        'initial_sigma' : np.linspace(0.01, 0.1, num=15),
        'mode' : ['1v1'],
        'adjust_for_base_rate' : [False]
    }  
}

