import copy
import numpy as np
import os
import pandas as pd
import datetime
from Factor import Factor
from config import root_fldr
from utils import normalize, winzorize

class Size(Factor):
  def __init__(self, startdate: pd.Timestamp, enddate: pd.Timestamp):
    Signals.__init__(self, target)
    self.save_fldr = f"{root_fldr}/data/descriptor/estimation"
    self.startdate = startdate
    self.enddate = enddate

  def calc(self):
    total_descriptor = pd.DataFrame()
    datadates = cal_util.dateSeq(startdate, enddate)

    for datadate in datadates:
      ma = datadate - pd.DateOffset(months=1)
      univ_cur = universe.loc[(universe['datadate'] <= datadate) & (universe['datadate'] >= ma)].reset_index(
          drop=True)

      ya = datadate - pd.DateOffset(years=1)
      fdmt_cur = fdmt.loc[(fdmt['rdq'] <= datadate) & (fdmt['datadate'] >= ya)].reset_index(drop=True)

      logmktcap = univ_cur.groupby('gvkey').apply(lambda x: pd.Series(dict(
          log_mktcap_21=np.log((x['cshoc'] * x['prccd']).mean())
      ))).reset_index()

      fmdt_qtr = fdmt_cur.groupby('gvkey')['datafqtr'].agg('max').reset_index()

      fdmt_cur = fdmt_cur.merge(fmdt_qtr, on=['gvkey', 'datafqtr'])
      asset = fdmt_cur.loc[fdmt_cur['atq'] > 0, ['gvkey', 'atq']]
      asset['log_asset'] = np.log(asset['atq'])

      size_des = pd.merge(logmktcap, asset[['gvkey', 'log_asset']], on='gvkey', how='outer')

      size_des['log_asset'] = winzorize(normalize(size_des['log_asset']))
      size_des['log_mktcap_21'] = winzorize(normalize(size_des['log_mktcap_21']))
      size_des = size_des.fillna(0)
      size_des['datadate'] = datadate

      total_descriptor = pd.concat([total_descriptor, size_des], ignore_index=True)
    total_descriptor.to_pickle(os.path.join(save_fldr, "size.pkl"))

  def getData(self):
    self.fdmt = pd.read_pickle(f"{root_fldr}/data/est_universe_fundq_us_raw_hist.pkl")
    self.universe = pd.read_pickle(
        f"{root_fldr}/data/est_universe_us_raw_hist_with_daily_return.pkl")
  