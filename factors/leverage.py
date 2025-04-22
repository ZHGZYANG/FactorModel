import copy
import numpy as np
import os
import pandas as pd
import datetime
from Factor import Factor
from config import root_fldr
from utils import normalize, winzorize

class Leverage(Factor):
  def __init__(self, startdate: pd.Timestamp, enddate: pd.Timestamp):
    Signals.__init__(self, target)
    self.save_fldr = f"{root_fldr}/data/descriptor/estimation"
    self.startdate = startdate
    self.enddate = enddate

  def calc(self):
    universe = pd.merge(universe, fdmt[['gvkey', 'datadate', 'apdedateq']], on=['gvkey', 'datadate'], how='left').sort_values(by=['gvkey', 'datadate'])
    universe[['apdedateq']] = universe.groupby('gvkey')[['apdedateq']].bfill().fillna(pd.to_datetime('today'))
    
    total_descriptor = pd.DataFrame()
    datadates = cal_util.dateSeq(startdate, enddate)

    for datadate in datadates:
      ya = cal_util.dateWrap(datadate, by=-300)
      univ_cur = universe.loc[(universe['apdedateq'] > ya) & (universe['datadate'] < datadate)].reset_index(drop=True)
      fdmt_cur = fdmt.loc[(fdmt['apdedateq'] > ya) & (fdmt['datadate'] < datadate)].reset_index(drop=True).drop_duplicates(subset=['gvkey', 'fyearq', 'fqtr'], keep='last')
      
      mktcap_avg = univ_cur.groupby('gvkey').apply(lambda x: (x['cshoc'] * x['prccd']).mean(), include_groups=False).reset_index(name='mktcap_avg')
      fdmt_cur['tncl'] = fdmt_cur['ltq'] - fdmt_cur['lctq']
      fdmt_cur = fdmt_cur.groupby('gvkey', group_keys=False).apply(lambda x: pd.Series({
          'tncl_avg': x['tncl'].mean(),
          'teqq_avg': x['teqq'].mean(),
          'ltq_avg': x['ltq'].mean(),
          'atq_avg': x['atq'].mean(),
      })).reset_index()
      leverage = pd.merge(fdmt_cur, mktcap_avg, on='gvkey', how='outer')

      leverage['blev'] = leverage['tncl_avg']/(leverage['tncl_avg'] + leverage['teqq_avg'])
      leverage['dtoa'] = leverage['ltq_avg']/leverage['atq_avg']
      leverage['mlev'] = leverage['tncl_avg']/(leverage['tncl_avg'] + leverage['mktcap_avg'])
      
      leverage['blev'] = winzorize(normalize(leverage['blev']))
      leverage['dtoa'] = winzorize(normalize(leverage['dtoa']))
      leverage['mlev'] = winzorize(normalize(leverage['mlev']))
      leverage = leverage.fillna(0)
      leverage['datadate'] = datadate
      total_descriptor = pd.concat([total_descriptor, leverage[['gvkey', 'blev', 'dtoa', 'mlev', 'datadate']]], ignore_index=True)
      print(f'{datadate:%Y%m%d} done')
    total_descriptor.to_pickle(os.path.join(save_fldr, "leverage.pkl"))

  def getData(self):
    self.fdmt = pd.read_pickle(f"{root_fldr}/data/est_universe_fundq_us_raw_hist.pkl")
    self.universe = pd.read_pickle(
        f"{root_fldr}/data/est_universe_us_raw_hist_with_daily_return.pkl")
  