"""
Implementations of online rating systems for paired comparison data.
One rating system per module, re-exported here so that
`from riix.models import Elo` works.

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
- `VSKF`: vector covariance simplified Kalman filter.
- `vElo`: variance incorporated Elo.
- `Melo`: multidimensional Elo for partially cyclic (rock-paper-scissors-like) games.
- `OnlineDiscDecomp`: online disc decomposition (Bertrand et al. 2023).
- `OnlineRaoKupper`: online gradient ascent on the Rao-Kupper draw model.
- `onlineMarkov`: online Markov method.
- `TemporalMassey`: temporalized Massey method.
- `Yuksel2024`: Cem Yuksel's 2024 matchmaking rating system.
- `AutogradRatingSystem`: gradient updates on arbitrary differentiable likelihoods.
- `BaselineRatingSystem`: win rate / appearance / random baselines.
"""

from riix.models.autograd_rating_system import AutogradRatingSystem
from riix.models.baselines import BaselineRatingSystem
from riix.models.constant_variance_glicko import ConstantVarianceGlicko
from riix.models.elo import Elo
from riix.models.elo_davidson import EloDavidson
from riix.models.elomentum import EloMentum
from riix.models.gen_elo import GenElo
from riix.models.glicko import Glicko
from riix.models.glicko2 import Glicko2
from riix.models.iterative_markov import onlineMarkov
from riix.models.melo import Melo
from riix.models.online_disc_decomp import OnlineDiscDecomp
from riix.models.online_rao_kupper import OnlineRaoKupper
from riix.models.skf import VSKF
from riix.models.temporal_massey import TemporalMassey
from riix.models.trueskill import TrueSkill
from riix.models.velo import vElo
from riix.models.weng_lin import WengLin
from riix.models.yuksel_2024 import Yuksel2024

__all__ = [
    'AutogradRatingSystem',
    'BaselineRatingSystem',
    'ConstantVarianceGlicko',
    'Elo',
    'EloDavidson',
    'EloMentum',
    'GenElo',
    'Glicko',
    'Glicko2',
    'Melo',
    'OnlineDiscDecomp',
    'OnlineRaoKupper',
    'TemporalMassey',
    'TrueSkill',
    'VSKF',
    'WengLin',
    'Yuksel2024',
    'onlineMarkov',
    'vElo',
]
