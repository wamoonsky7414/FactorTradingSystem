import pandas as pd
import numpy as np

from tabulate import tabulate
import plotly.graph_objects as go

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
                 start_time = '2013-07-01', 
                 end_time = '2024-03-01',
                 period_of_year:int = 252,
                 benchmark: pd.Series = pd.Series()):
        self.factor = factor
        self.expreturn = expreturn
        self.strategy = strategy
        self.buy_fee = buy_fee
        self.sell_fee = sell_fee
        self.start_time = pd.to_datetime(start_time)
        self.end_time = pd.to_datetime(end_time)
        self.benchmark = benchmark.loc[self.start_time:self.end_time]
        self.period_of_year = period_of_year
        self.weighting_by_period = None
        self.returns_by_period = None
        self.performace_report = None

    def backtest(self):
        self.returns_by_period = self.get_returns_by_period()
        self.summary_df = self.get_performance_report()
        print(tabulate(self.summary_df, headers='keys', tablefmt='pretty', showindex=True))
        self.get_pnl()
        return self.returns_by_period, self.summary_df
    
    def get_returns_by_period(self):
        self.weighting_by_period = self.get_weighting_by_period().loc[self.start_time:self.end_time]
        # self.expreturn.index = pd.to_datetime(self.expreturn.index)
        self.expreturn = self.expreturn.loc[self.start_time:self.end_time]
        total_fee_by_period = self.get_fee()
        profit_by_period = (self.weighting_by_period * self.expreturn).sum(axis=1)
        self.returns_by_period = profit_by_period - total_fee_by_period
        self.returns_by_period = self.returns_by_period.dropna()
        return self.returns_by_period
    
    def get_weighting_by_period(self):
        demean = self.factor.sub(self.factor.mean(axis=1), axis=0)
        self.weighting_by_period = demean.div(demean.abs().sum(axis=1), axis=0)

        if self.strategy == 'LS':
            self.weighting_by_period = self.weighting_by_period
        elif self.strategy == 'LO':
            self.weighting_by_period = self.weighting_by_period[self.weighting_by_period > 0] * 2
        elif self.strategy == 'SO':
            self.weighting_by_period = self.weighting_by_period[self.weighting_by_period < 0] * 2
        else:
            raise ValueError("Please use 'LS', 'LO', or 'SO' in strategy")
        # weighting_by_period.index = pd.to_datetime(weighting_by_period.index)
        return self.weighting_by_period

    def get_fee(self):
        self.weighting_by_period = self.weighting_by_period.loc[self.start_time:self.end_time]
        delta_weight = self.weighting_by_period.shift(1) - self.weighting_by_period
        buy_fees = delta_weight[delta_weight > 0]*(self.buy_fee)
        buy_fees = buy_fees.fillna(0)
        sell_fees = delta_weight.abs()[delta_weight < 0]*(self.sell_fee)
        sell_fees = sell_fees.fillna(0)
        # fee_by_period = buy_fees + sell_fees
        # delta_weight = weights.diff().abs()
        # buy_fees = delta_weight * self.buy_fee
        # sell_fees = delta_weight * self.sell_fee
        fee_by_period = buy_fees + sell_fees
        total_fee_by_period = fee_by_period.sum(axis=1)
        return total_fee_by_period


    # ==================================== get performances report =================================== #

    def get_performance_report(self):
        self.returns_by_period = self.returns_by_period.dropna()
        print('starttime:', self.returns_by_period.index[0], 'endtime: ',self.returns_by_period.index[-1])
        if not self.benchmark.empty:
            self.benchmark = self.benchmark.loc[self.returns_by_period.index[0]:self.returns_by_period.index[-1]]
            summary_df = pd.DataFrame({
                'Cumprod Total Returns': [f"{get_cumprod_returns(self.returns_by_period) * 100:.2f} %",
                                         f"{get_cumprod_returns(self.benchmark) * 100:.2f} %"],
                'Cumsum Total Returns': [f"{get_cumsum_returns(self.returns_by_period) * 100:.2f} %",
                                         f"{get_cumsum_returns(self.benchmark) * 100:.2f} %"],
                'Sharpe Ratio': [f"{get_sharpe(self.returns_by_period):.2f}",
                                 f"{get_sharpe(self.benchmark):.2f}"],
                'Annualized Ret': [f"{get_annual_returns(self.returns_by_period) * 100:.2f} %",
                                   f"{get_annual_returns(self.benchmark) * 100:.2f} %"],
                'Max Drawdown': [f"{get_mdd(self.returns_by_period) * 100:.2f} %",
                                 f"{get_mdd(self.benchmark) * 100:.2f} %"],
                'Volatility': [f"{get_volatility(self.returns_by_period) * 100:.2f} %",
                               f"{get_volatility(self.benchmark) * 100:.2f} %"],
                'STD': [f"{get_std(self.returns_by_period) * 100:.2f} %",
                        f"{get_std(self.benchmark) * 100:.2f} %"],
                'Turnover': [f"{get_turnover(self.weighting_by_period) * 100:.2f} %",
                             f"{np.nan}"]
            }, index=['Performance', 'Benchmark'])

        else:
            summary_df = pd.DataFrame({
                'Cumprod Total Returns': [f"{get_cumprod_returns(self.returns_by_period) * 100:.2f} %"],
                'Cumsum Total Returns': [f"{get_cumsum_returns(self.returns_by_period) * 100:.2f} %"],
                'Sharpe Ratio': [f"{get_sharpe(self.returns_by_period):.2f}"],
                'Annualized Ret': [f"{get_annual_returns(self.returns_by_period) * 100:.2f} %"],
                'Max Drawdown': [f"{get_mdd(self.returns_by_period) * 100:.2f} %"],
                'Volatility': [f"{get_volatility(self.returns_by_period) * 100:.2f} %"],
                'STD': [f"{get_std(self.returns_by_period) * 100:.2f} %"],
                'Turnover': [f"{get_turnover(self.weighting_by_period) * 100:.2f} %"]
            }, index=['Performance'])

        return summary_df

    # ======================= get plot =================== #
    def get_pnl(self):
        self.start_time = self.returns_by_period.index[0]
        self.end_time = self.returns_by_period.index[-1]
        benchmark_returns_filtered = self.benchmark.loc[self.start_time:self.end_time]
        benchmark_cumulative_returns = (1 + benchmark_returns_filtered).cumprod() - 1
        cumulative_returns = (1 + self.returns_by_period).cumprod() - 1

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns, mode='lines', name='Cumulative Returns'))
        fig.add_trace(go.Scatter(x=benchmark_cumulative_returns.index, y=benchmark_cumulative_returns, mode='lines', name='Benchmark', line=dict(color='#FFA500')))

        fig.update_layout(
            title='Cumulative Returns Over Time',
            xaxis_title='Date',
            yaxis_title='Cumulative Returns',
            width=750,
            height=450  
        )
        fig.show()

# ================================= get performs ==================================== #
def get_cumprod_returns(returns_by_period:pd.Series):
    ret_cum = (1 + returns_by_period).cumprod() - 1
    return ret_cum.iloc[-1]

def get_cumsum_returns(returns_by_period:pd.Series):
    ret_cum = returns_by_period.cumsum()
    return ret_cum.iloc[-1]

def get_sharpe(returns_by_period:pd.Series, period_of_year:int = 252):
    Sharpe_ratio = returns_by_period.mean() / returns_by_period.std() * np.sqrt(period_of_year)
    return Sharpe_ratio

def get_volatility(returns_by_period:pd.Series, period_of_year:int = 252):
    annual_vol = returns_by_period.std() * np.sqrt(period_of_year)
    return annual_vol

def get_std(returns_by_period:pd.Series):
    return returns_by_period.std()

def get_annual_returns(returns_by_period:pd.Series, period_of_year:int = 252):
    compound = (returns_by_period + 1).cumprod()
    days = len(compound)
    total_return = compound.iloc[-1] - 1
    annual_factor = period_of_year
    annualized_return = (total_return + 1) ** (annual_factor / days) - 1
    return annualized_return

def get_mdd(returns_by_period:pd.Series):
    compound = (returns_by_period + 1).cumprod()
    drawdowns = compound / compound.cummax() - 1
    max_drawdown = drawdowns.min()
    return abs(max_drawdown)

def get_turnover(weighting:pd.DataFrame):
    delta_weight = weighting.diff()
    daily_trading_value = delta_weight.abs().sum(axis=1)
    turnover = daily_trading_value.sum() / len(daily_trading_value)
    return turnover

def get_performance_report(returns_by_period:pd.Series, benchmark:pd.Series, period_of_year:int = 252):
    returns_by_period = returns_by_period.dropna()
    print('starttime:', returns_by_period.index[0], 'endtime: ',returns_by_period.index[-1])
    cumulative_returns = (1 + returns_by_period).cumprod() - 1

    if benchmark is not None and not benchmark.empty:
        benchmark = benchmark.loc[returns_by_period.index[0]:returns_by_period.index[-1]]
        summary_df = pd.DataFrame({
            'Cumprod Total Returns': [f"{get_cumprod_returns(returns_by_period) * 100:.2f} %",
                                        f"{get_cumprod_returns(benchmark) * 100:.2f} %"],
            'Cumsum Total Returns': [f"{get_cumsum_returns(returns_by_period) * 100:.2f} %",
                                        f"{get_cumsum_returns(benchmark) * 100:.2f} %"],
            'Sharpe Ratio': [f"{get_sharpe(returns_by_period, period_of_year):.2f}",
                                f"{get_sharpe(benchmark, period_of_year):.2f}"],
            'Annualized Ret': [f"{get_annual_returns(returns_by_period, period_of_year) * 100:.2f} %",
                                f"{get_annual_returns(benchmark, period_of_year) * 100:.2f} %"],
            'Max Drawdown': [f"{get_mdd(returns_by_period) * 100:.2f} %",
                                f"{get_mdd(benchmark) * 100:.2f} %"],
            'Volatility': [f"{get_volatility(returns_by_period, period_of_year) * 100:.2f} %",
                            f"{get_volatility(benchmark, period_of_year) * 100:.2f} %"],
            'STD': [f"{get_std(returns_by_period) * 100:.2f} %",
                    f"{get_std(benchmark) * 100:.2f} %"]
        }, index=['Performance', 'Benchmark'])

        print(tabulate(summary_df, headers='keys', tablefmt='pretty', showindex=True))

        benchmark_cumulative_returns = (1 + benchmark).cumprod() - 1

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns, mode='lines', name='Cumulative Returns'))
        fig.add_trace(go.Scatter(x=benchmark_cumulative_returns.index, y=benchmark_cumulative_returns, mode='lines', name='Benchmark', line=dict(color='#FFA500')))

        fig.update_layout(
            title='Cumulative Returns Over Time',
            xaxis_title='Date',
            yaxis_title='Cumulative Returns',
            width=750,
            height=450  
        )
        fig.show()        

    else:
        summary_df = pd.DataFrame({
            'Cumprod Total Returns': [f"{get_cumprod_returns(returns_by_period) * 100:.2f} %"],
            'Cumsum Total Returns': [f"{get_cumsum_returns(returns_by_period) * 100:.2f} %"],
            'Sharpe Ratio': [f"{get_sharpe(returns_by_period, period_of_year):.2f}"],
            'Annualized Ret': [f"{get_annual_returns(returns_by_period, period_of_year) * 100:.2f} %"],
            'Max Drawdown': [f"{get_mdd(returns_by_period) * 100:.2f} %"],
            'Volatility': [f"{get_volatility(returns_by_period, period_of_year) * 100:.2f} %"],
            'STD': [f"{get_std(returns_by_period) * 100:.2f} %"]
        }, index=['Performance'])

        print(tabulate(summary_df, headers='keys', tablefmt='pretty', showindex=True))


        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns, mode='lines', name='Cumulative Returns'))

        fig.update_layout(
            title='Cumulative Returns Over Time',
            xaxis_title='Date',
            yaxis_title='Cumulative Returns',
            width=750,
            height=450  
        )
        fig.show()


    return summary_df