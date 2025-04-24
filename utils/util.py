import numpy as np
import pandas as pd


def normalize(arr: pd.Series, mode: str = 'z-score') -> pd.Series:
  if mode == 'minmax':
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
  else:
    return (arr - np.nanmean(arr)) / np.nanstd(arr)


def winzorize(arr: pd.Series, range: int = 4) -> pd.Series:
  mean = np.nanmean(arr)
  std = np.nanstd(arr)
  return arr.clip(lower=mean - range * std, upper=mean + range * std)