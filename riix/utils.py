import numpy as np
import pandas as pd

def sigmoid(x, k=1):
    return 1 / (1 + np.exp(-x*k))

def get_num_competitors(df, competitor_cols):
    all_cols_unique = np.concatenate([df[col].unique() for col in competitor_cols])
    return len(set(all_cols_unique))