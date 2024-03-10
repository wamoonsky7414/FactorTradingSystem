import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, PROJECT_ROOT)

import pandas as pd
import numpy as np

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
        self.summary_df = None

    def weight(self):
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

        weight *= self.leverage
        return weight

    def fee(self):
        weights = self.weight().loc[self.start_time:self.end_time]
        delta_weight = weights.diff().abs()
        buy_fees = delta_weight * self.buy_fee
        sell_fees = delta_weight * self.sell_fee
        total_fee = buy_fees + sell_fees
        fee = total_fee.sum(axis=1)
        return dai_fee

    def backtest(self):
        daily_weight = self.weight().loc[self.start_time:self.end_time]
        self.expreturn = self.expreturn.loc[self.start_time:self.end_time]
        daily_fee = self.fee()
        daily_profit = (daily_weight * self.expreturn).sum(axis=1)
        self.daily_returns = daily_profit - daily_fee

        self.summary_df = self.summary()
        return self.daily_returns, self.summary_df

    def summary(self):
        summary_df = pd.DataFrame({
            'Sharpe Ratio': [self.sharpe()],
            'Annualized Ret': [f"{self.annual_returns() * 100}%"],
            'Max Drawdown': [f"{self.MDD() * 100}%"],
            'STD': [f"{self.std() * 100}%"],
            'Turnover': [f"{self.turnover() * 100}%"]
        }, index=['Performance'])
        return summary_df

    def sharpe(self):
        annual_factor = self.period_of_year
        Sharpe_ratio = round(self.daily_returns.mean() / self.daily_returns.std() * np.sqrt(annual_factor), 4)
        return Sharpe_ratio

    def std(self):
        return round(self.daily_returns.std(), 6)

    def annual_returns(self):
        compound = (self.daily_returns + 1).cumprod()
        days = len(compound)
        total_return = compound.iloc[-1] / compound.iloc[0] - 1
        annual_factor = self.period_of_year
        annualized_return = (total_return + 1) ** (annual_factor / days) - 1
        return round(annualized_return, 4)

    def turnover(self):
        delta_weight = self.weight().shift(1) - self.weight()
        daily_trading_value = delta_weight.abs().sum(axis=0)
        turnover = daily_trading_value.sum() / len(daily_trading_value)
        return round(turnover, 6)

    def MDD(self):
        compound = (self.daily_returns + 1).cumprod()
        drawdowns = []
        peak = compound[0]
        for price in compound:
            if price > peak:
                peak = price
            drawdown = (price - peak) / peak
            drawdowns.append(drawdown)
        max_drawdown = np.min(drawdowns)
        return round(abs(max_drawdown), 6)