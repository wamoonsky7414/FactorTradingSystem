import pandas as pd
import numpy as np

import warnings
with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)

# class Operator:
#     def __init__():

# operators_dict = dict(
#     ts_rank = ts_rank,    
# )

# arithmetic
def log(x: pd.DataFrame):
    return np.log(x[x!=0])

def sqrt(x: pd.DataFrame):
    return np.sqrt(x)

def sign(x: pd.DataFrame):
    return np.sign(x)

# time_series
def ts_rank(x: pd.DataFrame, d:int):
    return x.rolling(d).rank(pct=True)

def ts_corr(x: pd.DataFrame, y:pd.DataFrame, d:int):
    return x.rolling(d).corr(y)

# cross_section
def cs_rank(x: pd.DataFrame):
    return x.rank(axis=1, pct=True)

def cs_normalize(x: pd.DataFrame):
    row_means = x.mean(axis=1)
    row_stds = x.std(axis=1)
    normalized_df = (x.sub(row_means, axis=0)).div(row_stds, axis=0)
    return normalized_df

# ----------------------------- from Elija

def abs(x: pd.DataFrame) -> pd.DataFrame:
    return np.abs(x)#x.abs()

def delay(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.shift(d)

def correlation(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d).corr(y)#.replace([-np.inf, np.inf], 0).fillna(value=0)

def covariance(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d).cov(y)#.replace([-np.inf, np.inf], 0).fillna(value=0)

def cs_scale(x: pd.DataFrame, a:int=1) -> pd.DataFrame:
    return x.mul(a).div(x.abs().sum(axis = 1), axis='index')

def ts_delta(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.diff(d)

def ts_pct_change(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.pct_change(periods=d)

def signedpower(x: pd.DataFrame, a:int) -> pd.DataFrame:
    return x**a

def ts_decay_linear(x: pd.DataFrame, d:int) -> pd.DataFrame:
    # 過去 d 天的加權移動平均線，權重線性衰減 d, d ‒ 1, ..., 1（重新調整為總和為 1）
    result = x.values.copy()
    with np.errstate(all="ignore"):
        for i in range(1, d):
            result[i:] += (i+1) * x.values[:-i]
    result[:d] = np.nan
    return pd.DataFrame(result / np.arange(1, d+1).sum(),index = x.index,columns = x.columns)


def ts_min(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.rolling(d).min()

def ts_max(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.rolling(d).max()

def ts_argmin(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.rolling(d).apply(np.nanargmin, raw=True)+1

def ts_argmax(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.rolling(d).apply(np.nanargmax, raw=True)+1

def min(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return np.minimum(x,y)

def max(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return np.maximum(x,y)

def ts_sum(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.rolling(d).sum()

def ts_product(x: pd.DataFrame, d:int) -> pd.DataFrame:
    #return x.rolling(d, min_periods=d//2).apply(np.prod, raw=True)
    result = x.values.copy()
    with np.errstate(all="ignore"):
        for i in range(1, d):
            result[i:] *= x.values[:-i]
    return pd.DataFrame(result,index = x.index,columns = x.columns)

def ts_stddev(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.rolling(d).std()

def where(condition: pd.DataFrame, choiceA: pd.DataFrame, choiceB: pd.DataFrame) -> pd.DataFrame:
    condition_copy = pd.DataFrame(np.nan, index = condition.index, columns=condition.columns)
    condition_copy[condition] = choiceA
    condition_copy[~condition] = choiceB
    return condition_copy

def ts_mean(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.rolling(d).mean()

def ts_skewness(df: pd.DataFrame,period:int) -> pd.DataFrame:
    window_size = period
    rolling_skewness = df.rolling(window=window_size).skew()
    return rolling_skewness

def ts_kurtosis(df: pd.DataFrame,period:int) -> pd.DataFrame:
    window_size = period
    rolling_kurtosis = df.rolling(window=window_size).kurt()
    return rolling_kurtosis




def rolling_beta(df: pd.DataFrame, index_series:pd.Series ,period:int) -> pd.DataFrame:

    def ols_regression(X, Y):
        X = np.column_stack((np.ones(X.shape[0]), X))
        beta = np.linalg.inv(X.T @ X) @ X.T @ Y
        
        return beta[1:]

    def calculate_betas(stock_returns, index_series, window_length):
        betas = []
        indices = []  # 用於存儲對應的索引

        for i in range(window_length, len(stock_returns) + 1):
            X = index_series[i-window_length:i].values.reshape(-1, 1)
            Y = stock_returns[i-window_length:i].values
            beta = ols_regression(X, Y)
            betas.append(beta.item())  # 添加斜率系数
            indices.append(stock_returns.index[i-1])  # 添加对应的索引
            
        return pd.Series(betas, index=indices)


    market_ret = index_series.dropna()
    beta_df = pd.DataFrame()
    deleted_columns = []

    for target in df.columns:
        stock_ret = df[target]
        stock_ret.index = pd.to_datetime(stock_ret.index)
        if stock_ret.empty:
            deleted_columns.append(target)
        else:

            stock_first_index = stock_ret.index[0]
            market_first_index = market_ret.index[0]
            stock_last_index = stock_ret.index[-1]
            market_last_index = market_ret.index[-1]
            
            common_start_date = max(stock_first_index, market_first_index)
            common_end_date = min(stock_last_index, market_last_index)
            stock_ret = stock_ret[common_start_date:common_end_date]
            M_ret = market_ret.loc[common_start_date:common_end_date]  

            betas = calculate_betas(stock_ret, M_ret, period)
            betas = pd.DataFrame(betas)
            betas.columns = [target]
            beta_df = pd.concat([beta_df, betas], axis=1)
    
    return beta_df