import copy
import numpy as np
import os
import pandas as pd
import datetime
from Factor import Factor
from config import root_fldr
from util import normalize, winzorize

class Beta(Factor):
  def __init__(self, startdate: pd.Timestamp, enddate: pd.Timestamp):
    Signals.__init__(self, target)
    self.save_fldr = f"{root_fldr}/data/descriptor/estimation"
    self.startdate = startdate
    self.enddate = enddate

  def calc(self):
    sp500 = pd.read_pickle(f"{root_fldr}/crspm.pkl")[['sprtrn', 'caldt']].assign(caldt=lambda x: pd.to_datetime(x['caldt'])).set_index('caldt')

    total_descriptor = pd.DataFrame()
    datadates = cal_util.dateSeq(self.startdate, self.enddate)
    
    for datadate in datadates:
      ya2 = datadate - pd.DateOffset(years=2)
      univ_cur = self.universe.loc[(self.universe['datadate'] > ya2) & (self.universe['datadate'] < datadate)]
      idx_ret = sp500.loc[(sp500.index > ya2) & (sp500.index < datadate)]

      univ_cur = univ_cur[univ_cur.groupby('gvkey')['gvkey'].transform('size') > 40]
      wide_ret = univ_cur.pivot_table(index='datadate', columns='gvkey', values='ret').merge(idx_ret, left_index=True, right_index=True, how='left')
      
      covariance = wide_ret.cov()['sprtrn'].drop('sprtrn')
      idx_beta = covariance / np.square(wide_ret['sprtrn'].std())
      residual = wide_ret.loc[:, wide_ret.columns != 'sprtrn'] - np.dot(wide_ret['sprtrn'].to_frame(), idx_beta.to_frame().T)
      factor = (residual - residual.mean()).std(skipna=True).rename_axis('gvkey').reset_index(name='hsigma')
      
      idx_beta = idx_beta.rename_axis('gvkey').reset_index(name='beta')
      factor = factor.merge(idx_beta, on='gvkey', how='outer')
      factor['beta'] = winzorize(normalize(factor['beta']))
      factor['hsigma'] = winzorize(normalize(factor['hsigma']))
      factor = factor.fillna(0)
      factor['datadate'] = datadate
      total_descriptor = pd.concat([total_descriptor, factor], ignore_index=True)
      print(f'{datadate:%Y%m%d} done')
    total_descriptor.to_pickle(os.path.join(save_fldr, "beta.pkl"))
  def getData(self):
    self.fdmt = pd.read_pickle(f"{root_fldr}/data/est_universe_fundq_us_raw_hist.pkl")
    self.universe = pd.read_pickle(
        f"{root_fldr}/data/est_universe_us_raw_hist_with_daily_return.pkl")
  