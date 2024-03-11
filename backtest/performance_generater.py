import pandas as pd
import numpy as np

import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, PROJECT_ROOT)


class PerformanceGenerator(object):
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
        self.returns_by_period = None
        self.performace_report = None

    def weighting_by_strategy(self):
        demean = self.factor.sub(self.factor.mean(axis=1), axis=0)
        weight = demean.div(demean.abs().sum(axis=1), axis=0)

        if self.strategy == 'LS':
            weight = weight
        elif self.strategy == 'LO':
            weight = weight[weight > 0] * 2
        elif self.strategy == 'SO':
            weight = weight[weight < 0] * 2
        else:
            raise ValueError("Please use 'LS', 'LO', or 'SO' in strategy")
        return weight

    def get_fee(self):
        weights = self.weighting_by_strategy().loc[self.start_time:self.end_time]
        delta_weight = weights.diff().abs()
        buy_fees = delta_weight * self.buy_fee
        sell_fees = delta_weight * self.sell_fee
        fee_by_period = buy_fees + sell_fees
        total_fee_by_period = fee_by_period.sum(axis=1)
        return total_fee_by_period

    def backtest(self):
        weighting_by_period = self.weighting_by_strategy().loc[self.start_time:self.end_time]
        self.expreturn = self.expreturn.loc[self.start_time:self.end_time]
        total_fee_by_period = self.get_fee()
        profit_by_period = (weighting_by_period * self.expreturn).sum(axis=1)
        self.returns_by_period = profit_by_period - total_fee_by_period

        self.summary_df = self.performace_report()
        return self.returns_by_period, self.summary_df

    def get_performance_report(self):
        summary_df = pd.DataFrame({
            'Cumprod Total Returns': [f"{self.get_cumprod_returns().round(4) * 100} %"],
            'Cumsum Total Returns': [f"{self.get_cumsum_returns().round(4) * 100} %"],
            'Sharpe Ratio': [f"{self.get_sharpe().round(4) * 100} %"],
            'Annualized Ret': [f"{self.get_annual_returns().round(4) * 100} %"],
            'Max Drawdown': [f"{self.get_mdd().round(4) * 100} %"],
            'volatility': [f"{self.get_volatility().round(4) * 100} %"],
            'STD': [f"{self.get_std().round(4) * 100} %"],
            'Turnover': [f"{self.get_turnover().round(4) * 100} %"]
        }, index=['Performance'])
        return summary_df
    
    def get_cumprod_returns(self):
        ret_cum = (1 + self.returns_by_period).cumprod() -1
        return ret_cum

    def get_cumsum_returns(self):
        ret_cum = self.returns_by_period.cumsum()
        return ret_cum
    
    def get_sharpe(self):
        Sharpe_ratio = round(self.returns_by_period.mean() / self.returns_by_period.std() * np.sqrt(self.period_of_year), 4)
        return Sharpe_ratio

    def get_volatility(self):
        annual_vol =  self.returns_by_period.std() * np.sqrt(self.period_of_year)
        return annual_vol
    
    def get_std(self):
        return self.returns_by_period.std()


    def get_annual_returns(self):
        compound = (self.returns_by_period + 1).cumprod()
        days = len(compound)
        total_return = compound.iloc[-1] / compound.iloc[0] - 1
        annual_factor = self.period_of_year
        annualized_return = (total_return + 1) ** (annual_factor / days) - 1
        return annualized_return

    def get_turnover(self):
        delta_weight = self.weight().shift(1) - self.weight()
        daily_trading_value = delta_weight.abs().sum(axis=0)
        turnover = daily_trading_value.sum() / len(daily_trading_value)
        return turnover

    def get_mdd(self):
        compound = (self.daily_returns + 1).cumprod()
        drawdowns = []
        peak = compound[0]
        for price in compound:
            if price > peak:
                peak = price
            drawdown = (price - peak) / peak
            drawdowns.append(drawdown)
        max_drawdown = np.min(drawdowns)
        return abs(max_drawdown)