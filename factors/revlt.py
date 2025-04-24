import copy
import numpy as np
import os
import pandas as pd
import datetime
from Factor import Factor
from config import root_fldr
from util import normalize, winzorize

class Revlt(Factor):
  def __init__(self, startdate: pd.Timestamp, enddate: pd.Timestamp):
    Signals.__init__(self, target)
    self.save_fldr = f"{root_fldr}/data/descriptor/estimation"
    self.startdate = startdate
    self.enddate = enddate

  def calc(self):
    sp500 = pd.read_pickle(f"{root_fldr}/crspm.pkl")[['sprtrn', 'caldt']].assign(caldt=lambda x: pd.to_datetime(x['caldt'])).set_index('caldt')

    total_descriptor = pd.DataFrame()
    datadates = cal_util.dateSeq(self.startdate, self.enddate)

    self.universe['retflt'] = np.clip(self.universe['ret'], -0.99, 1)
    
    for datadate in datadates:
      ya4 = datadate - pd.DateOffset(years=4)
      univ_cur = self.universe.loc[(self.universe['datadate'] >= ya4) & (self.universe['datadate'] < datadate)]
      idx_ret = sp500.loc[(sp500.index >= ya4) & (sp500.index < datadate)]
      
      t_weight = pd.DataFrame({'datadate': univ_cur['datadate'].unique()}).sort_values(by='datadate').reset_index(drop=True)
      t_weight['weight'] = 2 ** ((t_weight.index) / 480)
      univ_cur = univ_cur.merge(t_weight, on='datadate', how='left')
      
      revlt = univ_cur.groupby('gvkey').apply(lambda x: -1e3 * np.sum(x['weight'] * np.log(1 + x['retflt'])) / np.sum(x['weight']) if len(x) > 100 else np.nan).dropna().reset_index(name='revlt')
      
      wide_ret = univ_cur.pivot_table(index='datadate', columns='gvkey', values='ret').merge(idx_ret, left_index=True, right_index=True, how='left')
      
      covariance = wide_ret.cov()['sprtrn'].drop('sprtrn')
      idx_beta = covariance / np.square(wide_ret['sprtrn'].std())
      residual = wide_ret.loc[:, wide_ret.columns != 'sprtrn'] - np.dot(wide_ret['sprtrn'].to_frame(), idx_beta.to_frame().T)
      alpha = (-residual.mean()).to_frame(name='alpha').rename_axis('gvkey').reset_index()
      
      revlt = revlt.merge(alpha, on='gvkey', how='outer')
      revlt['revlt'] = winzorize(normalize(revlt['revlt']))
      revlt['alpha'] = winzorize(normalize(revlt['alpha']))
      revlt = revlt.fillna(0)
      revlt['datadate'] = datadate
      total_descriptor = pd.concat([total_descriptor, revlt], ignore_index=True)
      print(f'{datadate:%Y%m%d} done')
    total_descriptor.to_pickle(os.path.join(save_fldr, "revlt.pkl"))

  def getData(self):
    self.fdmt = pd.read_pickle(f"{root_fldr}/data/est_universe_fundq_us_raw_hist.pkl")
    self.universe = pd.read_pickle(
        f"{root_fldr}/data/est_universe_us_raw_hist_with_daily_return.pkl")
  