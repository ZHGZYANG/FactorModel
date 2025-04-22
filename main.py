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

ROOT = os.getcwd()
sys.path.append(ROOT)
sys.path.append(f"{ROOT}/factors")
sys.path.append(f"{ROOT}/utils")

from backtest import portfolio_evaluation
from estimation import estimate
from factors import beta, leverage, momentum, rev1d, revlt, season, size, turnover, value, volatility
from preprocessing import *

class FactorPipeline:
    ROOT_DIR_DATASET = ''
    ROOT_DIR_SAVE = ''

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
        estu_univ()
        fdmt_data()
    
    def factor_calculate(self):
        for i, f in enumerate(self.Factors):
            f.getData()
            f.calc()
    
    def estimate(self):
        estimate()
    
    def backtest(self):
        result = portfolio_evaluation()
        print(result)

if __name__ == '__main__':
    startdate = pd.todatetime('2017-01-01')
    enddate = pd.todatetime('2024-01-01')
    calculator = FactorPipeline(startdate, enddate)
    
    calculator.preprocessing()
    calculator.factor_calculate()
    calculator.estimate()
    calculator.backtest()
