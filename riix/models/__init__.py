"""
Implementations of online rating systems for paired comparison data.
One rating system per module, re-exported here lazily so that
`from riix.models import Elo` works without importing every model module
(or optional dependencies like jax, which only AutogradRatingSystem needs).

Included rating systems:

- `Elo`: the classic Elo rating system.
- `EloDavidson`: Elo with Davidson's method of handling draws.
- `EloMentum`: Elo with momentum-based updates (heavy ball, Nesterov, or Adam).
- `GenElo`: Martin Ingram's Bayesian generalization of Elo.
- `ConstantVarianceGlicko`: Glicko with equal, unchanging variance for all competitors.
- `Glicko`: Mark Glickman's Glicko rating system.
- `Glicko2`: Mark Glickman's Glicko 2 rating system with volatility.
- `TrueSkill`: Microsoft's Bayesian TrueSkill rating system.
- `WengLin`: the Weng-Lin Bayesian online rating system (Bradley-Terry or Thurstone-Mosteller model).
- `WengLinThurstoneMosteller`: DEPRECATED, use WengLin with model='tm'.
- `VSKF`: vector covariance simplified Kalman filter.
- `vElo`: variance incorporated Elo.
- `Melo`: multidimensional Elo for partially cyclic (rock-paper-scissors-like) games.
- `OnlineDiscDecomp`: online disc decomposition (Bertrand et al. 2023).
- `OnlineRaoKupper`: online gradient ascent on the Rao-Kupper draw model.
- `onlineMarkov`: online Markov method.
- `TemporalMassey`: temporalized Massey method.
- `Yuksel2024`: Cem Yuksel's 2024 matchmaking rating system.
- `AutogradRatingSystem`: gradient updates on arbitrary differentiable likelihoods (requires the jax extra).
- `BaselineRatingSystem`: win rate / appearance / random baselines.
"""

import importlib

_MODEL_LOCATIONS = {
    'AutogradRatingSystem': 'autograd_rating_system',
    'BaselineRatingSystem': 'baselines',
    'ConstantVarianceGlicko': 'constant_variance_glicko',
    'Elo': 'elo',
    'EloDavidson': 'elo_davidson',
    'EloMentum': 'elomentum',
    'GenElo': 'gen_elo',
    'Glicko': 'glicko',
    'Glicko2': 'glicko2',
    'Melo': 'melo',
    'OnlineDiscDecomp': 'online_disc_decomp',
    'OnlineRaoKupper': 'online_rao_kupper',
    'TemporalMassey': 'temporal_massey',
    'TrueSkill': 'trueskill',
    'VSKF': 'skf',
    'WengLin': 'weng_lin',
    'WengLinThurstoneMosteller': 'weng_lin_thurstone_mosteller',
    'Yuksel2024': 'yuksel_2024',
    'onlineMarkov': 'iterative_markov',
    'vElo': 'velo',
}

__all__ = sorted(_MODEL_LOCATIONS)


def __getattr__(name):
    if name in _MODEL_LOCATIONS:
        module = importlib.import_module(f'.{_MODEL_LOCATIONS[name]}', __name__)
        return getattr(module, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(list(globals()) + __all__)
