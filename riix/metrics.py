import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def second_half_metric(metric, preds, scores):
    preds = preds[preds.shape[0]//2:]
    scores = scores[scores.shape[0]//2:]
    return metric(preds, scores)

def accuracy(preds, scores, thresh=0.5):
    correct = ((preds > thresh) & scores).sum()
    correct += ((preds < thresh) & (1-scores)).sum()
    correct += 0.5*(preds == thresh).sum()
    return correct / preds.shape[0]

def log_loss(preds, scores, eps=1e-15):
    preds = np.clip(preds, eps, 1-eps)
    return -((scores * np.log(preds)) + ((1-scores) * np.log(1-preds))).mean()

def brier_score(preds, scores):
    return np.square(preds - scores).mean()

is_higher_better_dict = {'accuracy' : True, 'log_loss' : False, 'brier_score' : False}
def is_better(x, y, metric):
    if is_higher_better_dict[metric]:
        return x > y
    elif not is_higher_better_dict[metric]:
        return y > x
    else:
        return False

def plot_calibration(preds, scores, n_bins=10):
    idxs = np.argsort(preds)
    preds_sorted = preds[idxs]
    scores_sorted = scores[idxs]
    pred_bins = np.split(preds_sorted, n_bins)
    score_bins = np.split(scores_sorted, n_bins)
    xs = [sb.mean() for sb in score_bins]
    ys = [pb.mean() for pb in pred_bins]
    plt.plot(xs, ys, '-o')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.show()

def all_metrics(preds, scores):
    metrics = {
        'accuracy' : accuracy(preds, scores),
        'log_loss' : log_loss(preds, scores),
        'brier_score' : brier_score(preds, scores),
        'second_half_accuracy' : second_half_metric(accuracy, preds, scores),
        'second_half_log_loss' : second_half_metric(log_loss, preds, scores),
        'second_half_brier_score' : second_half_metric(brier_score, preds, scores),
        'num' : len(preds)
    }
    return metrics

from datetime import datetime
year_segments = [f'timestamp.dt.year == 20{x}' for x in range(10,23)]
# year_segments = [
#     'timestamp.dt.year <= 2019',
#     'timestamp.dt.year == 2020',
#     'timestamp.dt.year == 2021',
#     'timestamp.dt.year == 2022',
# ]
region_segments = [f'region=="{x}"' for x in ['North America', 'Europe', 'China', 'Korea', 'International']]
region_segments += [x+'& level=="Primary"' for x in region_segments]
region_segments += [x+'& timestamp.dt.year == 2022' for x in region_segments]
level_segments = ['level=="Primary"', 'level=="Secondary"']
segments = year_segments + region_segments + level_segments


def segment_metrics(df, preds_col, scores_col, segments):
    results = {}
    results['overall'] = all_metrics(df[preds_col].values, df[scores_col].values)
    for segment in segments:
        segment_data = df.query(segment)
        metrics = all_metrics(segment_data[preds_col].values, segment_data[scores_col].values)
        results[segment] = metrics
    return results
