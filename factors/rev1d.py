import copy
import numpy as np
import os
import pandas as pd
import datetime
from Factor import Factor
from config import root_fldr
from utils import normalize, winzorize

class Rev1d(Factor):
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
      d20 = cal_util.dateWrap(datadate, by=-20)
      univ_cur = universe.loc[(universe['datadate'] >= d20) & (universe['datadate'] < datadate)]
      rev1d = univ_cur.groupby('gvkey')['retflt'].apply(lambda x: -np.exp(np.nansum(np.log1p(x))) - 1).reset_index(name='rev1d')
      rev1d['rev1d'] = winzorize(normalize(rev1d['rev1d']))
      rev1d = rev1d.fillna(0)
      rev1d['datadate'] = datadate
      total_descriptor = pd.concat([total_descriptor, rev1d], ignore_index=True)
    total_descriptor.to_pickle(os.path.join(save_fldr, "rev1d.pkl"))

  def getData(self):
    self.fdmt = pd.read_pickle(f"{root_fldr}/data/est_universe_fundq_us_raw_hist.pkl")
    self.universe = pd.read_pickle(
        f"{root_fldr}/data/est_universe_us_raw_hist_with_daily_return.pkl")
  