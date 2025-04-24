import datetime
import numpy as np
import os
import pandas as pd

from calendar_utils import Calendar_Util
from config import root_fldr

def load_rawdata(data_name: str):
  assert data_name in ('fundq', 'secd', 'ibes', 'secm'), f"unsupported data {data_name}"
  if data_name == 'ibes':
    detail_with_actuals_name = os.path.join(root_fldr, f"tr_ibes/det_epsint.pkl")
    price_target_name = os.path.join(root_fldr, f"tr_ibes/ptgdet.pkl")
    detail_with_actuals = pd.read_pickle(detail_with_actuals_name)
    price_target = pd.read_pickle(price_target_name)

    actuals_date_fields = ['pends', 'anndats', 'actdats']
    detail_with_actuals_date_fields = ['fpedats', 'anndats', 'actdats', 'revdats', 'actdats_act', 'anndats_act']
    price_target_date_fields = ['anndats', 'actdats']

    for d_f in detail_with_actuals_date_fields:
      if d_f in detail_with_actuals.columns:
        detail_with_actuals[d_f] = pd.to_datetime(detail_with_actuals[d_f])
    for d_f in price_target_date_fields:
      if d_f in price_target.columns:
        price_target[d_f] = pd.to_datetime(price_target[d_f])

    return detail_with_actuals, price_target
  else:
    file_name = os.path.join(root_fldr, f"comp_na_daily_all/{data_name}.pkl")
    data = pd.read_pickle(file_name)
    date_fields = ['datadate'] + ['apdedateq', 'fdateq', 'pdateq', 'rdq']
    for d_f in date_fields:
      if d_f in data.columns:
        data[d_f] = pd.to_datetime(data[d_f])
    return data

def gen_us_trading_calendar(save_fldr: str = f"{root_fldr}/reference"):
  secd = load_rawdata('secd')
  secd_ibm = secd.loc[secd['tic'] == 'IBM', ['datadate']].reset_index(drop=True)
  all_dates = pd.date_range(secd_ibm['datadate'].min(), secd_ibm['datadate'].max(), freq='D')
  calendar = pd.DataFrame({'calendar_date': all_dates}).sort_values('calendar_date').reset_index(drop=True)
  calendar['is_open'] = np.where(calendar['calendar_date'].isin(secd_ibm['datadate']), 1, 0)
  calendar['seq'] = calendar.index + 1
  if not os.path.exists(save_fldr): os.makedirs(save_fldr)
  calendar.to_pickle(os.path.join(save_fldr, 'calendar.pkl'))
  
  
def mkt_data():
  secd_raw = load_rawdata('secd')
  
  secd = secd_raw.loc[(secd_raw['fic'] == 'USA') & (secd_raw['curcdd'] == 'USD')]
  
  secd = secd.loc[secd['tpci'] == '0']
  
  secd = secd.loc[secd['secstat'] == 'A']
  
  secd['exchg'] = secd['exchg'].astype(int)
  secd = secd.loc[secd['exchg'].isin([11, 12, 14, 17])]
  
  prc_mask = (secd['prccd'] > 0) & (secd['prchd'] > 0) & (secd['prcld'] > 0) & (secd['prcod'] > 0)
  secd = secd.loc[prc_mask]
  
  issue_stats = secd.groupby(['gvkey', 'iid']).apply(lambda x: pd.Series(
    dict(adv=(x['prccd'] * x['cshtrd']).mean(),
         days=len(x)))).reset_index()
  
  issue_stats = issue_stats.sort_values(['gvkey', 'days', 'adv']).reset_index(drop=True)
  
  issue_stats_single = issue_stats.drop_duplicates(subset=['gvkey'], keep='last')
  
  secd = secd.merge(issue_stats_single[['gvkey', 'iid']], on=['gvkey', 'iid'])
  
  secd = secd.loc[(secd['cshtrd'] > 0) & (secd['cshoc'] > 0)]
  
  mktcap = secd.groupby(['gvkey', 'tic']).apply(lambda x: pd.Series(
    dict(mktcap_pct=(x['prccd'] * x['cshoc']).mean(),
         adv=(x['prccd'] * x['cshtrd']).mean(),
         days=len(x))), include_groups=False).reset_index()
  
  
  secd['adj_prccd'] = secd['prccd'] / secd['ajexdi']  # 
  
  secd = pd.merge(secd, mktcap, on=['gvkey', 'tic'], how='left')
  
  
  close_price = secd.sort_values(by=['datadate'])
  def daily_return(group: pd.DataFrame) -> pd.DataFrame:
    group['ret'] = group['adj_prccd'] / group['adj_prccd'].shift(1).fillna(method='bfill') - 1
    return group
  close_price = close_price.groupby('gvkey', group_keys=False).apply(daily_return)
  close_price['ret'].fillna(0, inplace=True)
  close_price.to_pickle(f"{root_fldr}/data/universe_us_raw_hist_with_daily_return.pkl")

  
def estu_univ(start_date: pd.Timestamp, end_date: pd.Timestamp):
  cal_util = Calendar_Util()
  data = pd.read_pickle(f"{root_fldr}/data/universe_us_raw_hist_with_daily_return.pkl")

  if not os.path.exists(f'{root_fldr}/data/daily/est_universe_us_raw_with_return/'):
    os.makedirs(f'{root_fldr}/data/daily/est_universe_us_raw_with_return/')
  

  datadates = cal_util.dateSeq(start_date, end_date)
  
  total_est_univ = pd.DataFrame()
  
  data['mktcap'] = data['prccd'] * data['cshoc']
  
  data['ym'] = data['datadate'].dt.to_period('M')
  data['trdval'] = data['prccd'] * data['cshtrd']
  
  stats = data.groupby(['gvkey', 'ym']).apply(lambda x: pd.Series(dict(
    days=len(x),
    median_trdval=np.median(x['trdval']),
    mktcap_last=x.iloc[-1]['mktcap']))).reset_index()
  
  stats['median_trdval_ratio'] = stats['median_trdval'] * stats['days'] / stats['mktcap_last']
  
  
  for datadate in datadates:
    daily_estu = data.loc[data['datadate'] == datadate]
  
    daily_estu = daily_estu.sort_values('mktcap', ascending=False).reset_index(drop=True)
    daily_estu['mktcap_pct'] = daily_estu['mktcap'] / daily_estu['mktcap'].sum()
    daily_estu['mktcap_pct_cumsum'] = daily_estu['mktcap_pct'].cumsum()
  
    mask = (daily_estu['mktcap_pct_cumsum'] <= 99 / 100) == True
    minimal_size = daily_estu.loc[mask, 'mktcap'].min()
  
    daily_estu_1 = daily_estu.loc[daily_estu['mktcap'] > 0.5 * minimal_size]
  
  
    ym_max = datadate.to_period('M')
    ym_min = (datadate - pd.DateOffset(months=13)).to_period('M')
    stats_12m = stats.loc[(stats['ym'] < ym_max) & (stats['ym'] >= ym_min)]
    atvr_12m = (stats_12m.groupby('gvkey')['median_trdval_ratio'].agg('mean') * 12).reset_index().rename(
      columns={'median_trdval_ratio': 'atvr12'})
  
    # atvr_3
    ym_min_3m = (datadate - pd.DateOffset(months=4)).to_period('M')
    stats_3m = stats.loc[(stats['ym'] < ym_max) & (stats['ym'] >= ym_min_3m)]
    atvr_3m = (stats_3m.groupby('gvkey')['median_trdval_ratio'].agg('mean') * 12).reset_index().rename(
      columns={'median_trdval_ratio': 'atvr3'})
  
    # another filter
    trd_days_cnt = stats_3m.groupby('gvkey')['days'].agg('sum').reset_index()
    max_days = trd_days_cnt.max()['days']
    trd_days_cnt['days_cnt_ratio'] = trd_days_cnt['days'] / max_days
    trd_days_cnt = trd_days_cnt.sort_values('days_cnt_ratio', ascending=True).reset_index(drop=True)
    
    univ_1 = atvr_12m.loc[atvr_12m['atvr12'] > 0.2]
    univ_2 = atvr_3m.loc[atvr_3m['atvr3'] > 0.2]
    univ_3 = trd_days_cnt.loc[trd_days_cnt['days_cnt_ratio'] > 0.9]
    univ_liquidity = pd.merge(univ_1, univ_2, on='gvkey', how='inner')
    univ_liquidity = pd.merge(univ_liquidity, univ_3, on='gvkey', how='inner')
    estu = pd.merge(univ_liquidity, daily_estu_1, on='gvkey', how='inner')
    estu.to_pickle(f'{root_fldr}/data/daily/est_universe_us_raw_with_return/{datadate:%Y%m%d}.pkl')
    total_est_univ = pd.concat([total_est_univ, estu], ignore_index=True)
    print(f'{datadate:%Y%m%d} done')
    
  total_est_univ.to_pickle(f"{root_fldr}/data/est_universe_us_raw_hist_with_daily_return.pkl")

  
def fdmt_data():
  fdmt_save_path = f"{root_fldr}/data/est_universe_fundq_us_raw_hist.pkl"
  
  universe = pd.read_pickle(f"{root_fldr}/data/est_universe_us_raw_hist_with_daily_return.pkl")
  company = pd.read_pickle(f'{root_fldr}/company.pkl')  # for GICS
  
  gvkey_list = list(universe['gvkey'].unique())
  fundq_raw = load_rawdata('fundq')
  fundq_raw = fundq_raw.loc[fundq_raw['gvkey'].isin(gvkey_list)].reset_index(drop=True)
  fundq_raw.groupby(['datafqtr'])['gvkey'].agg('count')
  fundq_raw.groupby(['rp', 'datafmt', 'acctstdq', 'costat', 'fic'])['gvkey'].agg('count')
  fundq_raw = fundq_raw.loc[fundq_raw['rp'] == 'Q']
  cnt = fundq_raw.groupby(['datafqtr', 'gvkey'])['gvkey'].agg('count')
  fundq_raw = fundq_raw.sort_values(['gvkey', 'datafqtr', 'rdq']).reset_index(drop=True)
  fundq_raw = fundq_raw.drop_duplicates(['datafqtr', 'gvkey'], keep='last').reset_index(drop=True)
  fundq_raw = fundq_raw.dropna(subset=['datafqtr'])
  
  GICS_keys = ['ggroup', 'gind', 'gsector', 'gsubind'] 
  fundq_raw = fundq_raw.merge(company[GICS_keys + ['gvkey']], on='gvkey', how='left')
  fundq_raw.to_pickle(fdmt_save_path)