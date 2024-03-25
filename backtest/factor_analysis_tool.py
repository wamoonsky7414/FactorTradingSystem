import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, PROJECT_ROOT)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
# import cufflinks
# cufflinks.go_offline()

from backtest.performance_generater import PerformanceGenerator

class FactorAnalysisTool(PerformanceGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def quantile_analysis(self, quantile_numbers=5):
        fig = go.Figure()
        print(f'Quantile {quantile_numbers} represents the highest factor value')

        daily_averages = {}
        weekly_averages = {}
        monthly_averages = {}

        for quantile in reversed(range(1, quantile_numbers + 1)):
            '''
            If a portfolio's return is negative and it corresponds to a lower quantile, 
            it indicates that short positions are effective.
            '''

            is_quantile = self.factor.rank(axis=1, pct=True) <= quantile / quantile_numbers
            is_quantile &= self.factor.rank(axis=1, pct=True) > (quantile - 1) / quantile_numbers
            
            self.weighting_by_period = self.get_weighting_by_period()

            quantile_weighting = self.weighting_by_period[is_quantile]
            delta_weight = quantile_weighting.shift(1) - quantile_weighting
            buy_fees = delta_weight[delta_weight > 0]*(self.buy_fee)
            buy_fees = buy_fees.fillna(0)
            sell_fees = delta_weight.abs()[delta_weight < 0]*(self.sell_fee)
            sell_fees = sell_fees.fillna(0)
            fee_by_period = buy_fees + sell_fees
            total_fee_by_period = fee_by_period.sum(axis=1)
            profit_by_period = (quantile_weighting * self.expreturn).sum(axis=1)
            quantile_returns_by_period = profit_by_period - total_fee_by_period
            quantile_returns_by_period = quantile_returns_by_period.dropna()
            cumulative_returns = (quantile_returns_by_period + 1).cumprod() -1

            fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns,
                        mode='lines', name=f'Quantile {quantile}'))
            
            # make average returns right here
            daily_avg = quantile_returns_by_period.mean() * 100
            weekly_avg = daily_avg * (self.period_of_year / 52)
            monthly_avg = daily_avg * (self.period_of_year / 12)
            
            daily_averages[quantile] = daily_avg
            weekly_averages[quantile] = weekly_avg
            monthly_averages[quantile] = monthly_avg

        if not self.benchmark.empty:
            self.benchmark = self.benchmark.loc[cumulative_returns.index[0]:cumulative_returns.index[-1]]
            cumprod_benchmark = (self.benchmark + 1).cumprod() -1
            cumprod_benchmark = cumprod_benchmark.loc[cumulative_returns.index[0]:cumulative_returns.index[-1]]
            fig.add_trace(go.Scatter(x=cumprod_benchmark.index, y=cumprod_benchmark,
                                    mode='lines', name=f'Benchmark'))  
            

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=list(daily_averages.keys()), y=list(daily_averages.values()),
                                name='Daily Average', marker_color='MediumBlue'))
        fig_bar.add_trace(go.Bar(x=list(weekly_averages.keys()), y=list(weekly_averages.values()),
                                name='Weekly Average', marker_color='DodgerBlue'))
        fig_bar.add_trace(go.Bar(x=list(monthly_averages.keys()), y=list(monthly_averages.values()),
                                name='Monthly Average', marker_color='LightSkyBlue'))

        fig_bar.update_layout(
            title='Quantile Performance: Daily, Weekly, Monthly Averages',
            xaxis_title='Quantile',
            yaxis_title='Average Returns (%)',
            yaxis=dict(tickformat=".2f"), 
            barmode='group', 
            legend_title="Time Frame",
            width=800,
            height=500
        )
        
        fig_bar.show()
        


        fig.update_layout(title='Quantile Performance Plot',
                        xaxis_title='Date',
                        yaxis_title='Cumulative Returns',
                        legend_title="Quantile",
                        width=800,
                        height=500)
        fig.show()
