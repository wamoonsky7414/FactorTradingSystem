import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, PROJECT_ROOT)

import pandas as pd
import numpy as np

from performance_generater import PerformanceGenerator

class FactorAnalysisTool(PerformanceGenerator):
    def __init__(self, 
                 factor: pd.DataFrame, 
                 expreturn: pd.DataFrame,
                 strategy='LS',  
                 buy_fee: float = 0.001425 * 0.3, 
                 sell_fee: float = 0.001425 * 0.3 + 0.003,
                 start_time='2012-01-01', 
                 end_time='2023-12-31',
                 period_of_year:int = 252):
        self.factor = factor
        self.expreturn = expreturn
        self.strategy = strategy
        self.buy_fee = buy_fee
        self.sell_fee = sell_fee
        self.start_time = start_time
        self.end_time = end_time
        self.period_of_year = period_of_year
        self.summary_df = None
