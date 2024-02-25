"""
Models Module
=============

This module contains implementations of various rating systems used to evaluate and compare the performance of competitors in games or sports. Each system provides a unique approach to calculating ratings, taking into account factors such as game outcomes, player volatility, and more.

Included Rating Systems:
- Elo: A simple and widely used rating system that adjusts player ratings based on match outcomes.
- Elomentum: Enhances the Elo system by incorporating momentum into rating adjustments.
- Glicko: An advanced system extending Elo by introducing rating deviation and volatility.
- Iterative Markov: Uses Markov assumptions for iterative rating updates.
- Melo: A variation of the Elo rating system with multidimensional representations.
- Online Disc Decomposition: A machine learning approach for rating updates.
- SKF: Implements simplified Kalman filters for rating estimation.
- Temporal Massey: A temporal adaptation of the Massey rating system.
- TrueSkill: A Bayesian rating system developed by Microsoft.
- Velo: A variant of the Elo system with variance.
- Weng-Lin Bradley-Terry: A Bayesian online rating system using the Bradley-Terry (logistic) model.
- Weng-Lin Thurstone-Mosteller: A Bayesian online rating system using the Thurstone-Mosteller (gaussian) model.

Each rating system is implemented as a class with methods for initializing the system, updating player ratings based on game outcomes, and calculating expected scores between players. These classes are designed for use in online rating systems where ratings are updated continuously as games are played.

This module also provides utility functions and constants used across different rating systems, such as mathematical functions for calculating expected scores and scaling factors.

"""
