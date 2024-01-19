Open source implementations of online rating systems focusing on efficiency for offline experimentation


## License
[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This package is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].
I've chosen a non-commercial license as overbroad protection to prevent the use of this package in the gambling and odds setting industries. If you want to use riix for your business in any other area please do not hesitate to reach out and I'll happily grant you an eternal lifetime license. :)

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

## terms
* Matchup: A single instance in which 2 or more entities compete. A generalization over a game, set, match, series, or round etc.
* Competitor: An entity which participates in a competition event. A generalzation over a player or a team. 
* N (int): number of matchups in a dataset
* C (int): number of unique competitors in a dataset
* competitor_id (int): ranging from 0 to C-1, a unique identifier for each competitor
* time_step (int): ranging from 0 to T. First competition in dataset is at time_step 0, last is at time_step T
* matchups (ndarray[int] of shape [N,2]): a matrix where each row represents a competition event. The The columns contain the competitor_ids of the two participants in the event
* time_steps (ndarray[int] of shape [N,]): a vector where the ith entry contains the integer time_step indicating when the ith matchup occured. Must be monotonically non-decreasing
* outcomes (ndarray[float] of shape [N,]): a vector where the ith entry contains the outcome of the ith competition event in the matchups array. 1.0 indicates the first competitor won, 0.0 indicaes the second competitor won, and 0.5 can indicate a draw

