import copy
import numpy as np
import os
import pandas as pd
import datetime
from Factor import Factor
from config import root_fldr
from util import normalize, winzorize

class Turnover(Factor):
  def __init__(self, startdate: pd.Timestamp, enddate: pd.Timestamp):
    Signals.__init__(self, target)
    self.save_fldr = f"{root_fldr}/data/descriptor/estimation"
    self.startdate = startdate
    self.enddate = enddate

  def calc(self):
    total_descriptor = pd.DataFrame()
    datadates = cal_util.dateSeq(self.startdate, self.enddate)
    if not os.path.exists(save_fldr): os.makedirs(save_fldr)
    for datadate in datadates:
      ma3 = datadate - pd.DateOffset(months=3)
      ma6 = datadate - pd.DateOffset(months=6)
      ya = datadate - pd.DateOffset(years=1)

      univ_cur = self.universe.loc[(self.universe['datadate'] <= datadate) & (self.universe['datadate'] >= ya)].reset_index(
          drop=True)

      univ_cur['ma3_flg'] = np.where(univ_cur['datadate'] >= ma3, 1, np.nan)
      univ_cur['ma6_flg'] = np.where(univ_cur['datadate'] >= ma6, 1, np.nan)

      turonver = univ_cur.groupby('gvkey', group_keys=False).apply(lambda x: pd.Series(dict(
          mndto3=(x['ma3_flg'] * x['cshtrd'] / x['cshoc']).mean(),
          mndto6=(x['ma6_flg'] * x['cshtrd'] / x['cshoc']).mean(),
          mndto12=(x['cshtrd'] / x['cshoc']).mean()
      ))).reset_index()

      turonver['mndto3'] = winzorize(normalize(turonver['mndto3']))
      turonver['mndto6'] = winzorize(normalize(turonver['mndto6']))
      turonver['mndto12'] = winzorize(normalize(turonver['mndto12']))
      turonver = turonver.fillna(0)
      turonver['datadate'] = datadate
      total_descriptor = pd.concat([total_descriptor, turonver], ignore_index=True)
      print(f'{datadate:%Y%m%d} done')
    total_descriptor.to_pickle(os.path.join(save_fldr, "turnover.pkl"))

  def getData(self):
    self.fdmt = pd.read_pickle(f"{root_fldr}/data/est_universe_fundq_us_raw_hist.pkl")
    self.universe = pd.read_pickle(
        f"{root_fldr}/data/est_universe_us_raw_hist_with_daily_return.pkl")
  