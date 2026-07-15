# riix vs other Python rating system packages

Fitting speed of riix — each model with `update_method='online'` (sequential per-match updates, the same semantics as the external packages' one-match-at-a-time loops, and the printed baseline) and `'batched'` (one vectorized update per rating period) — against other Python implementations of the same algorithms, on identical synthetic data with draws. Median of 3 runs after 1 warmup (which keeps the one-time jit compilation out of the timings — a first fit pays a ~0.2-1s compile per model and dataset shape), default hyperparameters, timing covers rating initialization plus processing every matchup. Speed only — this says nothing about predictive accuracy. Generated with:

```bash
uv run benchmarks/external.py --num-matchups 10000,100000 --num-competitors 100
```

Run on 2026-07-14 — AMD Ryzen 7 PRO 4750U (WSL2), Python 3.14.6, riix 0.1.0 at `bc5b680` (dirty, the release-prep tree), trueskill 0.4.5, openskill 6.2.0, glicko2 2.1.0, elote 1.1.0.

## 10,000 matchups, 100 competitors, 10 rating periods

| task | implementation | median seconds | time vs riix online |
| --- | --- | ---: | ---: |
| elo | **riix.Elo(online)** | 0.0044 | baseline |
|  | riix.Elo(batched) | 0.0024 | 0.55x |
|  | elote.EloCompetitor | 0.0185 | 4.24x |
| glicko | **riix.Glicko(online)** | 0.0121 | baseline |
|  | riix.Glicko(batched) | 0.0030 | 0.25x |
|  | elote.GlickoCompetitor | 0.0714 | 5.92x |
| glicko2 | **riix.Glicko2(online)** | 0.0347 | baseline |
|  | riix.Glicko2(batched) | 0.0030 | 0.09x |
|  | glicko2.Player | 0.2506 | 7.22x |
|  | elote.Glicko2Competitor | 0.1943 | 5.60x |
| trueskill | **riix.TrueSkill(online)** | 0.0228 | baseline |
|  | riix.TrueSkill(batched) | 0.0089 | 0.39x |
|  | trueskill.TrueSkill | 1.3625 | 59.67x |
| wenglin_bt | **riix.WengLin(bt, online)** | 0.0115 | baseline |
|  | riix.WengLin(bt, batched) | 0.0060 | 0.52x |
|  | openskill.BradleyTerryFull | 0.2378 | 20.75x |
| wenglin_tm | **riix.WengLin(tm, online)** | 0.0205 | baseline |
|  | riix.WengLin(tm, batched) | 0.0031 | 0.15x |
|  | openskill.ThurstoneMostellerFull | 0.2761 | 13.48x |

## 100,000 matchups, 100 competitors, 10 rating periods

| task | implementation | median seconds | time vs riix online |
| --- | --- | ---: | ---: |
| elo | **riix.Elo(online)** | 0.0185 | baseline |
|  | riix.Elo(batched) | 0.0092 | 0.50x |
|  | elote.EloCompetitor | 0.1634 | 8.84x |
| glicko | **riix.Glicko(online)** | 0.1100 | baseline |
|  | riix.Glicko(batched) | 0.0117 | 0.11x |
|  | elote.GlickoCompetitor | 0.6953 | 6.32x |
| glicko2 | **riix.Glicko2(online)** | 0.3129 | baseline |
|  | riix.Glicko2(batched) | 0.0095 | 0.03x |
|  | glicko2.Player | 2.4262 | 7.75x |
|  | elote.Glicko2Competitor | 1.7733 | 5.67x |
| trueskill | **riix.TrueSkill(online)** | 0.1862 | baseline |
|  | riix.TrueSkill(batched) | 0.0203 | 0.11x |
|  | trueskill.TrueSkill | 14.1514 | 75.99x |
| wenglin_bt | **riix.WengLin(bt, online)** | 0.0970 | baseline |
|  | riix.WengLin(bt, batched) | 0.0106 | 0.11x |
|  | openskill.BradleyTerryFull | 2.7079 | 27.92x |
| wenglin_tm | **riix.WengLin(tm, online)** | 0.1949 | baseline |
|  | riix.WengLin(tm, batched) | 0.0152 | 0.08x |
|  | openskill.ThurstoneMostellerFull | 3.1000 | 15.90x |

Reading the ratio column: the bold `riix.<Model>(online)` row is the baseline; larger is slower, smaller is faster. Ratios vs the external packages grow with dataset size because a warm riix fit is one compiled scan whose python overhead is constant, while the external packages pay python per match.

# Predictive accuracy

Prequential evaluation (each match predicted before it updates the ratings) of every riix model (at the float32, prequential defaults) and the external implementations on identical synthetic data, one rating period per match so all implementations see the same per-match semantics. Default hyperparameters everywhere, so differences reflect the algorithm *and* its defaults; metrics over all 10,000 matches. The seconds column includes riix's per-match rating-period overhead and is not the headline speed comparison (see the speed sections above). Generated with:

```bash
uv run benchmarks/accuracy.py
```

Run on 2026-07-14 — AMD Ryzen 7 PRO 4750U (WSL2), Python 3.14.6, riix 0.1.0 at `bc5b680` (dirty, the release-prep tree), trueskill 0.4.5, openskill 6.2.0, glicko2 2.1.0, elote 1.1.0.

## 10,000 matchups, 100 competitors, sorted by log loss

| implementation | log loss | accuracy | brier | seconds |
| --- | ---: | ---: | ---: | ---: |
| glicko2.Player | 0.5703 | 0.7027 | 0.1940 | 0.3169 |
| elote.Glicko2Competitor | 0.5703 | 0.7027 | 0.1940 | 0.2012 |
| riix.OnlineRaoKupper | 0.5724 | 0.6984 | 0.1950 | 0.0025 |
| elote.EloCompetitor | 0.5724 | 0.6985 | 0.1950 | 0.0326 |
| riix.Elo | 0.5724 | 0.6985 | 0.1950 | 0.0025 |
| riix.AutogradRatingSystem | 0.5724 | 0.6985 | 0.1950 | 0.0063 |
| riix.Melo | 0.5724 | 0.6985 | 0.1950 | 0.0079 |
| riix.GenElo | 0.5731 | 0.7027 | 0.1951 | 0.0043 |
| riix.ConstantVarianceGlicko | 0.5731 | 0.6976 | 0.1953 | 0.0035 |
| trueskill.TrueSkill | 0.5744 | 0.7040 | 0.1943 | 2.4366 |
| riix.EloDavidson | 0.5763 | 0.6972 | 0.1965 | 0.0036 |
| riix.EloMentum | 0.5822 | 0.6926 | 0.1986 | 0.0082 |
| riix.vElo | 0.5834 | 0.6956 | 0.1989 | 0.0085 |
| openskill.ThurstoneMostellerFull | 0.5843 | 0.6972 | 0.1982 | 0.3505 |
| riix.onlineMarkov | 0.5855 | 0.6953 | 0.1990 | 0.0025 |
| riix.TrueSkill | 0.5904 | 0.6929 | 0.2009 | 0.0200 |
| riix.TemporalMassey | 0.6026 | 0.6958 | 0.2069 | 0.0023 |
| riix.OnlineDiscDecomp | 0.6082 | 0.7041 | 0.2097 | 0.0070 |
| riix.WengLin | 0.6095 | 0.6825 | 0.2072 | 0.0246 |
| riix.Yuksel2024 | 0.6125 | 0.6899 | 0.2067 | 0.0128 |
| riix.Glicko2 | 0.6253 | 0.6692 | 0.2144 | 0.0322 |
| riix.BaselineRatingSystem | 0.6402 | 0.7015 | 0.2237 | 0.0068 |
| openskill.BradleyTerryFull | 0.6504 | 0.7006 | 0.2067 | 0.3210 |
| riix.Glicko | 0.6812 | 0.6511 | 0.2310 | 0.0121 |
| elote.GlickoCompetitor | 0.6930 | 0.7028 | 0.2498 | 0.0988 |
| riix.VSKF | 4.6099 | 0.6186 | 0.3706 | 0.0184 |

Reading notes: riix.Elo and elote.EloCompetitor agree to 4 decimals (same algorithm, same defaults — a correctness cross-check). Models whose time-dynamics defaults assume coarser rating periods (riix.Glicko's `c`, riix.Glicko2's cap, VSKF's Kalman dynamics) are penalized by the per-match granularity used here; tune hyperparameters before drawing conclusions about the algorithms themselves.
