## riix
Open source implementations of online rating systems. The focus is on efficiency for offline experimentation with static datasets, not for streaming/online applications.

## terms
* Competition event: A single instance in which 2 or more entities compete. A generalization over a game, set, match, series, or round etc.
* Competitor: An entity which participates in a competition event. A generalzation over a player or a team. 
* N (int): number of competition events in a dataset
* C (int): number of unique competitors in a dataset
* competitor_id (int): ranging from 0 to C-1, a unique identifier for each competitor
* time_step (int): ranging from 0 to T. First competition in dataset is at time_step 0, last is at time_step T
* schedule (ndarray[int] of shape [N,3]): a matrix where each row represents a competition event. The first column holds the time_step of the event. The second and third columns contain the competitor_ids of the participants in the event
* outcome (ndarray[float] of shape [N,]): a vector where the ith entry contains the outcome of the ith competition event in the schedule. 1.0 indicates the first competitor won, 0.0 indicaes the second competitor won, and 0.5 can indicate a draw

