import copy
import numpy as np
import os
import pandas as pd
import datetime
from Factor import Factor
from config import root_fldr
from utils import normalize, winzorize

class Season(Factor):
  def __init__(self, startdate: pd.Timestamp, enddate: pd.Timestamp):
    Signals.__init__(self, target)
    self.save_fldr = f"{root_fldr}/data/descriptor/estimation"
    self.startdate = startdate
    self.enddate = enddate

  def calc(self):
    total_descriptor = pd.DataFrame()
    datadates = cal_util.dateSeq(startdate, enddate)

    universe['retflt'] = np.clip(universe['ret'], -0.99, 1)

    for datadate in datadates:
      season_return = pd.DataFrame()
      for year in range(1, 6):
        ya = datadate - pd.DateOffset(years=year)
        univ_cur = universe.loc[(universe['datadate'] >= ya) & (universe['datadate'] < cal_util.dateWrap(ya, by=20))]
        return_by_year = univ_cur.groupby('gvkey')['retflt'].apply('sum').reset_index(name='season_return')
        return_by_year['year'] = year
        season_return = pd.concat([season_return, return_by_year], ignore_index=True)

      season = season_return.groupby('gvkey')['season_return'].apply('mean').reset_index(name='season')

      season['season'] = winzorize(normalize(season['season']))
      season['datadate'] = datadate
      total_descriptor = pd.concat([total_descriptor, season], ignore_index=True)
      print(f'{datadate:%Y%m%d} done')
    total_descriptor.to_pickle(os.path.join(save_fldr, "season.pkl"))

  def getData(self):
    self.fdmt = pd.read_pickle(f"{root_fldr}/data/est_universe_fundq_us_raw_hist.pkl")
    self.universe = pd.read_pickle(
        f"{root_fldr}/data/est_universe_us_raw_hist_with_daily_return.pkl")
  