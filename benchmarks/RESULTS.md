# riix vs other Python rating system packages

Fitting speed of riix (`update_method='online'`, the same sequential per-match semantics as the external packages) against other Python implementations of the same algorithms, on identical synthetic data with draws. Median of 3 runs after 1 warmup, default hyperparameters, timing covers rating initialization plus processing every matchup. Speed only — this says nothing about predictive accuracy. Generated with:

```bash
uv run benchmarks/external.py --num-matchups 10000,100000 --num-competitors 100
```

Run on 2026-07-03 — AMD Ryzen 7 PRO 4750U (WSL2), Python 3.14.6, riix at `4a238ba` (dirty), trueskill 0.4.5, openskill 6.2.0, glicko2 2.1.0, elote 1.1.0.

## 10,000 matchups, 100 competitors, 10 rating periods

| task | implementation | median seconds | time vs riix |
| --- | --- | ---: | ---: |
| elo | **riix.Elo** | 0.0028 | baseline |
|  | elote.EloCompetitor | 0.0181 | 6.40x |
| glicko | **riix.Glicko** | 0.0449 | baseline |
|  | elote.GlickoCompetitor | 0.0611 | 1.36x |
| glicko2 | **riix.Glicko2** | 0.0713 | baseline |
|  | glicko2.Player | 0.2615 | 3.67x |
|  | elote.Glicko2Competitor | 0.1880 | 2.64x |
| trueskill | **riix.TrueSkill** | 0.0488 | baseline |
|  | trueskill.TrueSkill | 1.2564 | 25.76x |
| wenglin_bt | **riix.WengLin(bt)** | 0.0372 | baseline |
|  | openskill.BradleyTerryFull | 0.2229 | 6.00x |
| wenglin_tm | **riix.WengLin(tm)** | 0.0480 | baseline |
|  | openskill.ThurstoneMostellerFull | 0.2965 | 6.18x |

## 100,000 matchups, 100 competitors, 10 rating periods

| task | implementation | median seconds | time vs riix |
| --- | --- | ---: | ---: |
| elo | **riix.Elo** | 0.0410 | baseline |
|  | elote.EloCompetitor | 0.1866 | 4.55x |
| glicko | **riix.Glicko** | 0.4579 | baseline |
|  | elote.GlickoCompetitor | 0.6733 | 1.47x |
| glicko2 | **riix.Glicko2** | 0.6983 | baseline |
|  | glicko2.Player | 2.5480 | 3.65x |
|  | elote.Glicko2Competitor | 1.8109 | 2.59x |
| trueskill | **riix.TrueSkill** | 0.4386 | baseline |
|  | trueskill.TrueSkill | 12.2705 | 27.98x |
| wenglin_bt | **riix.WengLin(bt)** | 0.4052 | baseline |
|  | openskill.BradleyTerryFull | 2.2741 | 5.61x |
| wenglin_tm | **riix.WengLin(tm)** | 0.4770 | baseline |
|  | openskill.ThurstoneMostellerFull | 2.6291 | 5.51x |

riix's vectorized `update_method='batched'` mode (one update per rating period, not shown here because its within-period semantics differ from the sequential loops above) is a further 20–50x faster than riix online; run `benchmarks/external.py` to see both.
