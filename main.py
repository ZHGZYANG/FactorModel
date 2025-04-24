import argparse
import copy
import logging
from multiprocessing import Process
import numpy as np
import os
import pandas as pd
import pickle
import sys
import warnings
import wrds

ROOT = os.getcwd()
sys.path.append(ROOT)
sys.path.append(f"{ROOT}/factors")
sys.path.append(f"{ROOT}/utils")

from backtest import portfolio_evaluation
from estimation import estimate
from factors import beta, leverage, momentum, rev1d, revlt, season, size, turnover, value, volatility
from preprocessing import *

class FactorPipeline:
    def __init__(self, startdate: pd.Timestamp, enddate: pd.Timestamp):
        self.startdate = startdate
        self.enddate = enddate
        self.Factors = [
            beta.Beta(self.startdate, self.enddate), 
            leverage.Leverage(self.startdate, self.enddate), 
            momentum.Momentum(self.startdate, self.enddate), 
            rev1d.Rev1d(self.startdate, self.enddate), 
            revlt.revlt(self.startdate, self.enddate), 
            season.Season(self.startdate, self.enddate), 
            size.Size(self.startdate, self.enddate), 
            turnover.Turnover(self.startdate, self.enddate), 
            value.Value(self.startdate, self.enddate), 
            volatility.Volatility(self.startdate, self.enddate),
        ]

    def preprocessing(self):
        mkt_data()
        estu_univ(self.startdate, self.enddate)
        fdmt_data()
    
    def factor_construction(self):
        for i, f in enumerate(self.Factors):
            f.getData()
            f.calc()
    
    def estimate(self):
        estimate(self.startdate, self.enddate)
    
    def backtest(self):
        result = portfolio_evaluation(self.startdate, self.enddate)

def download_data(start_date: str, end_date: str):
    db = wrds.Connection(wrds_username='', wrds_password='')  # replace to your username
    query = f"""
        SELECT *
        FROM comp_na_daily_all.fundq
        WHERE datadate >= '{start_date}' AND datadate <= '{end_date}'
    """ 
    fundamentals_quarterly = db.raw_sql(query).to_pickle('data/comp_na_daily_all/fundq.pkl')
    
    query = f"""
        SELECT *
        FROM comp_na_daily_all.secd
        WHERE datadate >= '{start_date}' AND datadate <= '{end_date}'
    """ 
    db.raw_sql(query).to_pickle('data/comp_na_daily_all/secd.pkl')
    db.close()



if __name__ == '__main__':
    startdate = '2023-01-01'
    enddate = '2023-03-01'
    
    download_data(startdate, enddate)
    
    calculator = FactorPipeline(pd.to_datetime(startdate), pd.to_datetime(enddate))
    
    calculator.preprocessing()
    calculator.factor_construction()
    calculator.estimate()
    calculator.backtest()
