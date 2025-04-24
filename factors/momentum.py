import copy
import numpy as np
import os
import pandas as pd
import datetime
from Factor import Factor
from config import root_fldr
from util import normalize, winzorize

class Momentum(Factor):
  def __init__(self, startdate: pd.Timestamp, enddate: pd.Timestamp):
    Signals.__init__(self, target)
    self.save_fldr = f"{root_fldr}/data/descriptor/estimation"
    self.startdate = startdate
    self.enddate = enddate

  def calc(self):
    total_descriptor = pd.DataFrame()
    datadates = cal_util.dateSeq(self.startdate, self.enddate)

    for datadate in datadates:
      ya1 = datadate - pd.DateOffset(years=1)
      ya2 = datadate - pd.DateOffset(years=2)
      univ_cur = self.universe.loc[(self.universe['datadate'] > ya2) & (self.universe['datadate'] < datadate)]
      univ_cur['ya1_flg'] = np.where(univ_cur['datadate'] > ya1, 1, np.nan)
      
      momentom = univ_cur.groupby('gvkey', group_keys=False).apply(lambda x: pd.Series(dict(
          rstr12=(x['ya1_flg'] * np.log1p(x['ret'].fillna(0))).sum(),
          rstr24=np.log1p(x['ret'].fillna(0)).sum()
      ))).reset_index()    
      
      momentom['rstr12'] = winzorize(normalize(momentom['rstr12']))
      momentom['rstr24'] = winzorize(normalize(momentom['rstr24']))
      momentom = momentom.fillna(0)
      momentom['datadate'] = datadate
      total_descriptor = pd.concat([total_descriptor, momentom], ignore_index=True)
      print(f'{datadate:%Y%m%d} done')
    total_descriptor.to_pickle(os.path.join(save_fldr, "momentom.pkl"))

  def getData(self):
    self.fdmt = pd.read_pickle(f"{root_fldr}/data/est_universe_fundq_us_raw_hist.pkl")
    self.universe = pd.read_pickle(
        f"{root_fldr}/data/est_universe_us_raw_hist_with_daily_return.pkl")
  