import os

import matplotlib.pyplot as plt
import datetime
from functools import partial
import numpy as np
import pickle
import pandas as pd
import scipy.optimize as optimize
import seaborn as sns
from sklearn.linear_model import LinearRegression

from config import root_fldr

def objective(beta: np.array, X: np.ndarray, y: np.array, stock_count: int) -> np.ndarray:
    residuals = np.dot(X, beta) - y
    return np.sum(residuals ** 2)


def constraint(beta: np.array, weight: np.array, factors_amount: int) -> np.float64:
    assert beta.shape[0] == weight.shape[0] + factors_amount, (beta.shape, weight.shape)
    subset_sum = np.sum(beta[factors_amount:] * weight)
    return subset_sum


def compute_industry_cap_weight(universe: pd.DataFrame, datadate: datetime.datetime) -> pd.DataFrame:
    ind = pd.read_pickle(f'{root_fldr}/company.pkl')

    ma = datadate - pd.DateOffset(months=1)
    universe = universe.loc[(universe['datadate'] <= datadate) & (universe['datadate'] >= ma)].reset_index(drop=True)

    universe_with_ind = universe.merge(ind[['gvkey', 'gind']], on='gvkey', how='left')
    cap_weight = universe_with_ind[['mktcap', 'gind', 'datadate']].groupby(['datadate', 'gind']).sum()
    one_month_weight = cap_weight / cap_weight.groupby('datadate').sum()

    return one_month_weight.groupby('gind').mean()


def compute_market_return(universe: pd.DataFrame) -> pd.DataFrame:
    universe['mktcap'] = universe['mktcap'] / universe['mktcap'].sum()
    market_return = sum(universe['ret'] * universe['mktcap'])
    return market_return
    
def estimate():
    universe = pd.read_pickle(f"{root_fldr}/data/est_universe_us_raw_hist_with_daily_return.pkl")
    ind = pd.read_pickle(f'{root_fldr}/data/Industry_allocation/industry_allocation.pkl')
    start_date = pd.to_datetime('2018-01-01')
    end_date = pd.to_datetime('2023-12-29')
    datadates = cal_util.dateSeq(start_date, end_date)

    FACTORS = ['size', 'turnover', 'value', 'volatility', 'season', 'rev1d', 'momentom', 'leverage', 'revlt', 'beta','value',]
    DESCRIPTORS_UNIV = {factor: pd.read_pickle(f'{root_fldr}/data/descriptor/estimation/{factor}.pkl') for factor in FACTORS}
    FACTOR_COEFS = {key: pd.read_pickle(f'{root_fldr}/data/risk_index_formulation/estimation/{key}.pkl') for key in FACTORS}
    FACTORS_RETURN = pd.DataFrame(columns=FACTORS + ['market_factor', 'market_ret', 'datadate'])
    FACTORS_EXPOSURE = pd.DataFrame(columns=['gvkey', 'exposure', 'factor', 'datadate'])

    for datadate in datadates:
        descriptors = {factor: DESCRIPTORS_UNIV[factor][DESCRIPTORS_UNIV[factor]['datadate'] == datadate] for factor in FACTORS}
        
        for factor in FACTORS:
            descriptors[factor][f"{factor}_exposure"] = 0
            descriptor_list = FACTOR_COEFS[factor].columns if isinstance(next(iter(FACTOR_COEFS.items()))[1], pd.DataFrame) else FACTOR_COEFS[factor].keys()
            for descriptor in descriptor_list:
                descriptors[factor][f"{factor}_exposure"] += descriptors[factor][descriptor] * FACTOR_COEFS[factor][descriptor]
        
        joined_df = pd.DataFrame(columns=['gvkey'])
        for factor in FACTORS:
            how = 'outer' if factor == FACTORS[0] else 'inner'
            joined_df = joined_df.merge(descriptors[factor][['gvkey', f'{factor}_exposure']], on='gvkey', how=how)
        
        ret = universe[universe['datadate'] == datadate][['gvkey', 'ret', 'mktcap']]
        ret['ret'] = ret['ret'].shift(-1).ffill()
        joined_df = joined_df.merge(ret, on='gvkey')    

        for factor in FACTORS:
            FACTORS_EXPOSURE = pd.concat([FACTORS_EXPOSURE, joined_df[['gvkey', f'{factor}_exposure']].assign(
                factor=factor, datadate=datadate).rename(columns={f'{factor}_exposure': 'exposure'})], ignore_index=True)

        joined_df.set_index('gvkey', inplace=True)
        joined_df = pd.merge(joined_df, ind, how='left', left_index=True, right_index=True)

        cap_weight = []
        cap_weight_df = compute_industry_cap_weight(universe.loc[universe['datadate'] == datadate], datadate)

        for column in joined_df.columns:
            if column.startswith('gind_'):
                code = column.split('_')[1]
                if code in cap_weight_df.index:
                    cap_weight.append(float(cap_weight_df.loc[code].iloc[0]))
                else:
                    cap_weight.append(0)
        cap_weight = np.array(cap_weight)

        X = joined_df.drop(['ret', 'mktcap'], axis=1).values
        y = joined_df['ret'].values

        beta_initial = np.zeros(X.shape[1])
        wrapped_constraint = partial(constraint, weight=cap_weight, factors_amount=len(FACTORS))
        cons: dict = {'type': 'eq', 'fun': lambda x: wrapped_constraint(x)}
        wrapped_objective = partial(objective, X=X, y=y, stock_count=joined_df.shape[0])

        result = optimize.minimize(wrapped_objective, x0=beta_initial, constraints=cons)
        factor_ret = [result.x[i] for i in range(len(FACTORS))] + [(np.dot(X, result.x) - y).mean(),
                      compute_market_return(universe.loc[universe['datadate'] == datadate]), datadate]

        FACTORS_RETURN = pd.concat([FACTORS_RETURN, pd.DataFrame([factor_ret], columns=FACTORS_RETURN.columns)], ignore_index=True)
    FACTORS_EXPOSURE.to_pickle(os.path.join(f'{root_fldr}/data/', 'factors_exposure_value.pkl'))
    FACTORS_RETURN.to_pickle(os.path.join(f'{root_fldr}/data/', 'factors_return_value.pkl'))