import numpy as np
from dataclasses import dataclass

@dataclass
class RatingDataset:
    time_steps: np.array    # dtype: np.int32, shape: (N,)
    competitors: np.array   # dtype: np.int32, shape: (N,2)
    outcomes: np.array      # dtype: np.float64, shape: (N,)
