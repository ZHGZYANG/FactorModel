import copy
import numpy as np
import os
import pandas as pd
import datetime
from Factor import Factor
from config import root_fldr
from utils import normalize, winzorize

class Value(Factor):
  def __init__(self, startdate: pd.Timestamp, enddate: pd.Timestamp):
    Signals.__init__(self, target)
    self.save_fldr = f"{root_fldr}/data/descriptor/estimation"
    self.startdate = startdate
    self.enddate = enddate

  def calc(self):
    universe = pd.merge(universe, fdmt[['gvkey', 'datadate','apdedateq']], on=['gvkey', 'datadate'], how='left').sort_values(by=['gvkey', 'datadate'])
    universe[['apdedateq']] = universe.groupby('gvkey')[['apdedateq']].bfill().fillna(pd.to_datetime('today'))

    total_descriptor = pd.DataFrame()
    datadates = cal_util.dateSeq(startdate, enddate)
    if not os.path.exists(save_fldr): os.makedirs(save_fldr)
    
    for datadate in datadates:
      univ_cur = universe.loc[universe['datadate'] == datadate].reset_index(drop=True)
      fdmt_cur = fdmt.groupby('gvkey').last().reset_index()
      univ_cur = pd.merge(univ_cur, fdmt_cur, on='gvkey')

      btop = univ_cur[['gvkey', 'cshoc', 'prccd', 'ceqq']]
      btop['btop'] = btop['ceqq'] / (btop['cshoc'] * btop['prccd'])

      ya1 = cal_util.dateWrap(datadate, by=-300)
      ya5 = datadate - pd.DateOffset(years=5)
      univ_cur = universe.loc[(universe['apdedateq'] > ya5) & (universe['datadate'] < datadate)].reset_index(drop=True)
      univ_cur['ya1_flg'] = np.where(univ_cur['datadate'] >= ya1, 1, np.nan)
      fdmt_cur = fdmt.loc[(fdmt['apdedateq'] > ya5) & (fdmt['datadate'] < datadate)].reset_index(drop=True)
      fdmt_cur['ya1_flg'] = np.where(fdmt_cur['apdedateq'] >= ya1, 1, np.nan)
      
      univ_cur = univ_cur.groupby('gvkey', group_keys=False).apply(lambda x: pd.Series(dict(
          mktcap_avg_1 = np.nanmean(x['ya1_flg'] * x['cshoc'] * x['prccd']),
          mktcap_avg_5 = (x['cshoc'] * x['prccd']).mean(),
      ))).reset_index()
      fdmt_cur = fdmt_cur.groupby('gvkey', group_keys=False).apply(lambda x: pd.Series(dict(
          chechy_sum_1 = np.nansum(x['ya1_flg'] * x['chechy']),
          niq_sum_1 = np.nansum(x['ya1_flg'] * x['niq']),
          chechy_sum_5 = x['chechy'].sum(),
          niq_sum_5 = x['niq'].sum(),
      ))).reset_index()
      
      
      value = pd.merge(univ_cur, fdmt_cur, on='gvkey', how='outer')
      # cash
      value['ctop'] = value['chechy_sum_1'] / value['mktcap_avg_1']
      value['ctop5'] = (value['chechy_sum_5'] / 5) / value['mktcap_avg_5']
      # earn
      value['etop'] = value['niq_sum_1'] / value['mktcap_avg_1']
      value['etop5'] = (value['niq_sum_5'] / 5) / value['mktcap_avg_5']

      value = pd.merge(btop[['gvkey', 'btop']], value[['gvkey', 'ctop', 'ctop5', 'etop', 'etop5']], on='gvkey', how='outer')
      value['btop'] = winzorize(normalize(value['btop']))
      value['ctop'] = winzorize(normalize(value['ctop']))
      value['ctop5'] = winzorize(normalize(value['ctop5']))
      value['etop'] = winzorize(normalize(value['etop']))
      value['etop5'] = winzorize(normalize(value['etop5']))
      value = value.fillna(0)
      value['datadate'] = datadate
      total_descriptor = pd.concat([total_descriptor, value], ignore_index=True)
    total_descriptor.to_pickle(os.path.join(save_fldr, "value.pkl"))

  def getData(self):
    self.fdmt = pd.read_pickle(f"{root_fldr}/data/est_universe_fundq_us_raw_hist.pkl")
    self.universe = pd.read_pickle(
        f"{root_fldr}/data/est_universe_us_raw_hist_with_daily_return.pkl")
  