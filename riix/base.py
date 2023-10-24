from abc import ABC, abstractmethod
import numpy as np
from riix.utils.data_utils import RatingDataset

class BaseRatingSystem(ABC):
    
    @abstractmethod
    def fit(Rat)