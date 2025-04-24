import copy
import numpy as np
import os
import pandas as pd
import datetime
from Factor import Factor
from config import root_fldr
from util import normalize, winzorize

class Volatility(Factor):
  def __init__(self, startdate: pd.Timestamp, enddate: pd.Timestamp):
    Signals.__init__(self, target)
    self.save_fldr = f"{root_fldr}/data/descriptor/estimation"
    self.startdate = startdate
    self.enddate = enddate

  def calc(self):
    total_descriptor = pd.DataFrame()
    datadates = cal_util.dateSeq(self.startdate, self.enddate)

    for datadate in datadates:
      # CMRA calculation (last year)
      ya = datadate - pd.DateOffset(years=1)
      univ_cur = self.universe.loc[(self.universe['datadate'] <= datadate) & (self.universe['datadate'] > ya)].reset_index(drop=True)

      univ_cur['cumret'] = univ_cur.groupby('gvkey')['ret'].transform(lambda x: (1 + x).cumprod())
      cmra = univ_cur.groupby('gvkey').apply(lambda x: pd.Series(dict(
          cmra=np.log(x['cumret'].max()) - np.log(x['cumret'].min())
      ))).reset_index()
      cmra['cmra'].fillna(0, inplace=True)
      cmra['cmra'] = winzorize(normalize(cmra['cmra']))

      # DHILO calculation (last 3 months)
      mth3 = datadate - pd.DateOffset(months=3)
      prcdata = self.universe.loc[(self.universe['datadate'] <= datadate) & (self.universe['datadate'] > mth3)].reset_index(
          drop=True)

      dhilo = prcdata.groupby('gvkey').apply(lambda x: pd.Series(dict(
          dhilo=np.median(np.log(x['prchd']) - np.log(x['prcld']))
      ))).reset_index()
      dhilo['dhilo'].fillna(0, inplace=True)
      dhilo['dhilo'] = winzorize(normalize(dhilo['dhilo']))

      # DVRAT calculation (last 2 years)
      ya2 = datadate - pd.DateOffset(years=2)
      retdata = self.universe.loc[(self.universe['datadate'] <= datadate) & (self.universe['datadate'] > ya2)].reset_index(drop=True)

      # DVRAT Calculation
      retdata['cumretq'] = retdata.groupby('gvkey')['ret'].transform(
          lambda x: x.rolling(window=10, min_periods=1).sum())
      dvrat = retdata.groupby('gvkey').apply(lambda x: pd.Series(dict(
          dvrat=(np.mean(x['cumretq'] ** 2) / np.mean(x['ret'] ** 2)) / 10 - 1
      ))).reset_index()
      dvrat['dvrat'].fillna(0, inplace=True)
      dvrat['dvrat'] = winzorize(normalize(dvrat['dvrat']))

      # Merge descriptors on 'gvkey'
      descriptor_df = pd.merge(cmra[['gvkey', 'cmra']], dhilo[['gvkey', 'dhilo']], on='gvkey', how='outer')
      descriptor_df = pd.merge(descriptor_df, dvrat[['gvkey', 'dvrat']], on='gvkey', how='outer')
      descriptor_df.fillna(0, inplace=True)
      descriptor_df['datadate'] = datadate
      total_descriptor = pd.concat([total_descriptor, descriptor_df], ignore_index=True)
    total_descriptor.to_pickle(os.path.join(save_fldr, "volatility.pkl"))

  def getData(self):
    self.fdmt = pd.read_pickle(f"{root_fldr}/data/est_universe_fundq_us_raw_hist.pkl")
    self.universe = pd.read_pickle(
        f"{root_fldr}/data/est_universe_us_raw_hist_with_daily_return.pkl")
  