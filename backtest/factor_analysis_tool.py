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
        for quantile in reversed(range(1, quantile_numbers + 1)):

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

            # quantile_returns = self.expreturn[is_quantile].mean(axis=1).dropna()
            # cumulative_returns = (quantile_returns + 1).cumprod() -1

            # 添加至圖表
            fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns,
                                    mode='lines', name=f'Quantile {quantile}'))

        if not self.benchmark.empty:
            cumprod_benchmark = (self.benchmark + 1).cumprod() -1
            cumprod_benchmark = cumprod_benchmark.loc[cumulative_returns.index[0]:cumulative_returns.index[-1]]
            fig.add_trace(go.Scatter(x=cumprod_benchmark.index, y=cumprod_benchmark,
                                    mode='lines', name=f'Benchmark'))  

        fig.update_layout(title='Quantile Performance Plot',
                        xaxis_title='Date',
                        yaxis_title='Cumulative Returns',
                        legend_title="Quantile")
        fig.show()

