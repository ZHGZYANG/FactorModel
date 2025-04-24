import cvxportfolio as cvx
import numpy as np
import pandas as pd
import mosek
import os
import yfinance

from config import root_fldr


def get_factor_covariance(save: bool = False) -> pd.DataFrame:
    FACTORS_RETURN = pd.read_pickle(os.path.join(f'{root_fldr}/data/', 'factors_return.pkl')).drop(
        ['market_factor', 'market_ret'], axis=1).set_index('datadate')
    ewm_cov = FACTORS_RETURN.ewm(halflife=504, adjust=True).cov(pairwise=True)
    ewm_cov.index.set_names(['datadate', 'factor'], inplace=True)
    ewm_cov = ewm_cov.tz_localize('UTC', level=0)
    if save:
        ewm_cov.to_pickle(f'{root_fldr}/data/factors_covariance.pkl')
    return ewm_cov


def get_factors_exposure(portfolio: list) -> pd.DataFrame:
    FACTORS_EXPOSURE = pd.read_pickle(os.path.join(f'{root_fldr}/data/', 'factors_exposure.pkl'))
    exposures = FACTORS_EXPOSURE.pivot_table(index=['datadate', 'factor'], columns='gvkey', values='exposure',
                                             sort=True)
    exposures = exposures.tz_localize('UTC', level=0)
    return exposures[portfolio]


def get_idyosyncratic_variance(exposures: pd.DataFrame, portfolio_returns: pd.DataFrame):
    FACTORS_RETURN = pd.read_pickle(os.path.join(f'{root_fldr}/data/', 'factors_return.pkl')).drop(
        ['market_factor', 'market_ret'], axis=1).set_index('datadate')
    exposures = exposures.reset_index().melt(id_vars=['datadate', 'factor'], var_name='gvkey', value_name='exposure')
    factors_ret = FACTORS_RETURN.tz_localize('UTC').reset_index().melt(id_vars='datadate', var_name='factor',
                                                                       value_name='ret')
    explained_ret = pd.merge(exposures, factors_ret, on=['datadate', 'factor'])
    explained_ret['ret'] *= explained_ret['exposure']
    explained_ret = explained_ret.pivot_table(index='datadate', columns='gvkey', values='ret')
    stock_specific_ret = (portfolio_returns - explained_ret)
    return stock_specific_ret.rolling(window=2).var().bfill().abs()


def get_bias_statistics(portfolio_weights: pd.DataFrame, portfolio_returns: pd.DataFrame,
                        idyosyncratic_variance: pd.DataFrame, exposures: pd.DataFrame,
                        factors_covariance: pd.DataFrame) -> np.float64:
    portfolio_idyosyncratic_variance = idyosyncratic_variance.reset_index().melt(id_vars='datadate', var_name='gvkey',
                                                                                 value_name='idyosyncratic_variance').merge(
        portfolio_weights, on='gvkey', how='left')
    portfolio_idyosyncratic_variance['portfolio_variance'] = portfolio_idyosyncratic_variance['weight'] ** 2 * \
                                                             portfolio_idyosyncratic_variance['idyosyncratic_variance']
    portfolio_variance = portfolio_idyosyncratic_variance.groupby('datadate')['portfolio_variance'].apply('sum')

    portfolio_exposure = exposures.reset_index().melt(id_vars=['datadate', 'factor'], var_name='gvkey', value_name='exposure').merge(portfolio_weights, on='gvkey', how='left')
    portfolio_exposure['exposure'] *= portfolio_exposure['weight']
    portfolio_exposure['exposure'] = portfolio_exposure.groupby(['datadate', 'factor'])['exposure'].transform('sum')
    portfolio_exposure = portfolio_exposure.drop_duplicates(subset=['datadate', 'factor']).drop(
        columns=['gvkey', 'weight'])
    portfolio_exposure = portfolio_exposure.set_index(['datadate', 'factor'])
    portfolio_exposure = portfolio_exposure.pivot_table(index='datadate', columns='factor', values='exposure')
    portfolio_cov = factors_covariance.loc[portfolio_exposure.index.get_level_values('datadate')]

    portfolio_variance += pd.Series(
        [portfolio_exposure.loc[date].values.T @ portfolio_cov.loc[date].values @ portfolio_exposure.loc[date].values
         for date in portfolio_exposure.index], index=portfolio_exposure.index)

    weights = pd.Series(portfolio_weights.set_index('gvkey')['weight'])
    portfolio_returns = (portfolio_returns * weights).sum(axis=1)
    standardized_return = (portfolio_returns.shift(1).ffill() / portfolio_variance).replace([np.inf, -np.inf], np.nan)
    return standardized_return.std()
    
def backtest(exposures: pd.DataFrame | None, factors_covariance: pd.DataFrame | None,
                   idyosyncratic_variance: pd.DataFrame | None, portfolio_returns: pd.DataFrame, 
                   portfolio_volumes: pd.DataFrame, portfolio_prices: pd.DataFrame,
                   start_time: pd.Timestamp, end_time: pd.Timestamp, GAMMA=1, KAPPA=0, baseline=False):
    if baseline:
        objective = cvx.ReturnsForecast() - GAMMA * (cvx.FullCovariance() + KAPPA * cvx.RiskForecastError()) - cvx.StocksTransactionCost()
    else:
        objective = cvx.ReturnsForecast() - GAMMA * (cvx.FactorModelCovariance(F=exposures, Sigma_F=factors_covariance,
                    d=idyosyncratic_variance) + KAPPA * cvx.RiskForecastError()) - cvx.StocksTransactionCost()

    constraints = [cvx.LeverageLimit(1)]

    policy = cvx.SinglePeriodOptimization(objective, constraints)
    simulator = cvx.MarketSimulator(returns=portfolio_returns, volumes=portfolio_volumes, prices=portfolio_prices,
                                    datasource=cvx.data.SymbolData, base_location='./tmp2',
                                    cash_key="USDOLLAR", trading_frequency=None)
    return simulator.backtest(policy, start_time=start_time, end_time=end_time)
    
def init(start_date: pd.Timestamp, end_date: pd.Timestamp, portfolio: list = None) -> tuple:
    universe = pd.read_pickle(f"{root_fldr}/data/est_universe_us_raw_hist_with_daily_return.pkl")[
        ['datadate', 'gvkey', 'ret', 'cshtrd', 'prccd', 'prcod']]
    universe = universe.loc[(universe['datadate'] <= end_date) & (universe['datadate'] >= start_date)].reset_index(
        drop=True)
    universe.set_index(['datadate', 'gvkey'], inplace=True)
    universe.sort_index(level=['datadate', 'gvkey'], inplace=True)
        
    if portfolio:
        returns = universe.pivot_table(index='datadate', columns='gvkey', values='ret').tz_localize('UTC')[portfolio]
        volumes = universe.pivot_table(index='datadate', columns='gvkey', values='cshtrd').tz_localize('UTC')[portfolio]
        prices = universe.pivot_table(index='datadate', columns='gvkey', values='prccd').tz_localize('UTC')[portfolio]
    else:
        returns = universe.pivot_table(index='datadate', columns='gvkey', values='ret').tz_localize('UTC').dropna(
            how='any', axis=1)
        volumes = universe.pivot_table(index='datadate', columns='gvkey', values='cshtrd').tz_localize('UTC')[
            returns.columns]
        prices = universe.pivot_table(index='datadate', columns='gvkey', values='prccd').tz_localize('UTC')[
            returns.columns]
    return returns, volumes, prices
    
def portfolio_evaluation(portfolio: list=None, weight: list=None,  start_date: pd.Timestamp, end_date: pd.Timestamp):
    if not portfolio:
        portfolio = ['001690', '160329', '012141']
        weight = [0.2, 0.3, 0.5]
    returns, volumes, prices = init(start_time, end_time, portfolio)
    exposures = get_factors_exposure(portfolio)
    covariance = get_factor_covariance()
    idyosyncratic_variance = get_idyosyncratic_variance(exposures, returns)

    portfolio_weights = pd.DataFrame(dict(gvkey=portfolio, weight=weight))
    bias_statistic = get_bias_statistics(portfolio_weights, returns, idyosyncratic_variance, exposures, covariance)

    model_result = backtest(exposures, covariance, idyosyncratic_variance, returns, volumes, prices, start_time,
                                  end_time, GAMMA=1, KAPPA=0)
    baseline_result = backtest(None, None, None, returns, volumes, prices, start_time, end_time, GAMMA=1, KAPPA=0, baseline=True)
    return bias_statistic, model_result.annualized_average_return, model_result.annualized_volatility, baseline_result.annualized_average_return, baseline_result.annualized_volatility
