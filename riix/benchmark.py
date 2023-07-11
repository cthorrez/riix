from riix.datasets import DatasetFactory
from riix.elo import EloV2
from riix.gradient.opper import Opperate
from riix.skf.skf_glicko import Glicko
from riix.skf.skf_trueskill import TrueSkill
from riix.skf.skf import vSFK
from riix.utils import get_num_competitors
from riix.metrics import all_metrics


def load_1v1_datasets():
    sc2_dataset = DatasetFactory.load_dataset(data_path='../data/sc2_matches_5-27-2023.csv', game='starcraft2')
    print(sc2_dataset.info())
    melee_dataset = DatasetFactory.load_dataset(data_path='../data/melee_matches_5-28-2023.csv', game='melee')
    print(melee_dataset.info())
    rl_dataset = DatasetFactory.load_dataset(data_path='../data/rl_matches.csv', game='rl')
    print(rl_dataset.info())
    lol_dataset = DatasetFactory.load_dataset(data_path='../data/lol_games.csv', game='lol', level='team')
    print(lol_dataset.info())
    dota2_dataset = DatasetFactory.load_dataset(data_path='../data/dota2_games.csv', game='dota2', level='team')
    print(dota2_dataset.info())

    datasets = {
        'sc2' : sc2_dataset,
        'melee' : melee_dataset,
        'rl' : rl_dataset,
        'lol' : lol_dataset,
        'dota2' : dota2_dataset
    }
    return datasets


def run_1v1_benchmark():
    datasets = load_1v1_datasets()

    models = {
        'elo' : EloV2,
        'glicko' : Glicko,
        'trueskill' : TrueSkill,
        'vSKF' : vSFK,
        'opper' : Opperate,
    }

    for game, dataset in datasets.items():

        dataset['days'] = (dataset['timestamp'] - dataset.iloc[0]['timestamp']).dt.days
        dataset = dataset.drop('timestamp', axis=1)

        num_competitors = get_num_competitors(df=dataset, competitor_cols=dataset.columns[:2])

        for model_name, model_class in models.items():
            model = model_class(num_competitors=num_competitors)
            rows = list(dataset.itertuples(index=False, name=None))
            probs = model.run_schedule(rows)
            # model.topk(5)
            metrics = all_metrics(probs, dataset['score'])
            for metric, val in metrics.items():
                print(f'{game:<7}{model_name:<10}{metric:<16}: {val:.6f}')




if __name__ == '__main__':
    run_1v1_benchmark()