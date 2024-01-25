Open source implementations of online rating systems focusing on efficiency for offline experimentation

## When to use riix
This package is designed to accelerate experiments studying and comparing rating systems. In the scenario where you have paired comparison datasets with a known number of competitors and time range, riix exploits that information to achieve fast runtimes. It's **not** useful in the streaming case where new data with new competitors are coming in. It also only currently supports 1v1 competitions but future support for two team competitions is planned. (more than 2 teams is not)


> I have a large dataset of player matches for a game and want to determine which rating system out of Elo, Glicko, TrueSkill etc. gives the best predictive accuracy.

Use riix! 👍

> I want to incorporate skill based matchmaking into the game I am creating and want a package to compute ratings for players on the fly.

There are lots of other great python packages for that too! (just not riix)
* [openskill.py](https://github.com/OpenDebates/openskill.py) Multi-way competitions with [Weng-Lin](https://www.jmlr.org/papers/v12/weng11a.html) rating systems
* [trueskill](https://github.com/topics/trueskill) Open source implementation of Microsoft's rating system
* [PythonSkills](https://github.com/agoragames/PythonSkills) A port of [Moserware's](https://www.moserware.com/2010/03/computing-your-skill.html) Glicko and TrueSkill [C# code](https://github.com/moserware/Skills) to python
* [skelo](https://github.com/mbhynes/skelo/tree/main) Elo and Glicko 2 with the [scikit-learn](https://scikit-learn.org/stable/) interface
* [mmr-python](https://github.com/kari/mmr-python) Glicko and Weng-Lin rating systems
* glicko2
  * [sublee/glicko2](https://github.com/sublee/glicko2)
  * [deepy/glicko2](https://github.com/deepy/glicko2)
* [whole_history_rating](https://github.com/pfmonville/whole_history_rating) Python port of [WHR](https://www.remi-coulom.fr/WHR/)

## Example
```
import numpy as np
from riix.models.elo import Elo
from riix.utils import MatchupDataset, generate_matchup_data
from riix.metrics import binary_metrics_suite

df = generate_matchup_data() # replace with your pandas dataframe
dataset = MatchupDataset(
    df,
    competitor_cols=['competitor_1', 'competitor_2'],
    outcome_col='outcome',
    datetime_col='date',
    rating_period='1D',
)

>>> loaded dataset with:
>>> 1000 matchups
>>> 100 unique competitors
>>> 10 rating periods of length 1D

model = Elo(num_competitors=dataset.num_competitors)
probs = []
for matchups, outcomes, time_step in dataset:
    probs.extend(model.predict(matchups))
    model.update(matchups, outcomes)
metrics = binary_metrics_suite(probs=np.array(probs), outcomes=dataset.outcomes)

>>> {'accuracy': 0.799,
>>> 'log_loss': 0.5363807117304283,
>>> 'brier_score': 0.1292005977180302,
>>> 'duration': 0.003565073013305664}

model.print_top_k(k=5, competitor_names=dataset.competitors)

>>> competitor   	rating
>>> competitor_74	1730.661227
>>> competitor_48	1727.693338
>>> competitor_47	1725.183076
>>> competitor_89	1719.935741
>>> competitor_62	1712.183214
```

## License
This package is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].
I've chosen a non-commercial license as overbroad protection to prevent the use of this package in the gambling and odds setting industries. If you would like to use riix for your business in any other area please do not hesitate to reach out and I'll happily grant you an eternal lifetime license. :)

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/

## Abut the name
The name riix represents an attempt to cleverly represent the idea of "R8" (pronounced "rate") alphabetically using the Roman numeral IIX in place of 8. By the time I realized the correct numeral would have been VIII I was already attatched to the name riix so I stuck with it. However on further research it turns out the Romans themselves occasionally used this form as well! [Why Romans Sometimes Wrote 8 as VIII, And Sometimes as IIX: A Possible Explanation](https://scholarworks.utep.edu/cs_techrep/1555/)

