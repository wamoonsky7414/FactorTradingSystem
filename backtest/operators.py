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


# def rolling_beta(df: pd.DataFrame, index_series:pd.Series ,period:int) -> pd.DataFrame:

#     def ols_regression(X, Y):
#         X = np.column_stack((np.ones(X.shape[0]), X))
#         beta = np.linalg.inv(X.T @ X) @ X.T @ Y
        
#         return beta[1:]

#     def calculate_betas(stock_returns, index_series, window_length):
#         betas = []
#         indices = []  # 用於存儲對應的索引

#         for i in range(window_length, len(stock_returns) + 1):
#             X = index_series[i-window_length:i].values.reshape(-1, 1)
#             Y = stock_returns[i-window_length:i].values
#             beta = ols_regression(X, Y)
#             betas.append(beta.item())  # 添加斜率系数
#             indices.append(stock_returns.index[i-1])  # 添加对应的索引
            
#         return pd.Series(betas, index=indices)


#     market_ret = index_series.dropna()
#     beta_df = pd.DataFrame()
#     deleted_columns = []

#     for target in df.columns:
#         stock_ret = df[target]
#         stock_ret.index = pd.to_datetime(stock_ret.index)
#         if stock_ret.empty:
#             deleted_columns.append(target)
#         else:

#             stock_first_index = stock_ret.index[0]
#             market_first_index = market_ret.index[0]
#             stock_last_index = stock_ret.index[-1]
#             market_last_index = market_ret.index[-1]
            
#             common_start_date = max(stock_first_index, market_first_index)
#             common_end_date = min(stock_last_index, market_last_index)
#             stock_ret = stock_ret[common_start_date:common_end_date]
#             M_ret = market_ret.loc[common_start_date:common_end_date]  

#             betas = calculate_betas(stock_ret, M_ret, period)
#             betas = pd.DataFrame(betas)
#             betas.columns = [target]
#             beta_df = pd.concat([beta_df, betas], axis=1)
    
#     return beta_df


# def rolling_alpha(df: pd.DataFrame, index_series:pd.Series ,period:int) -> pd.DataFrame:

#     def ols_regression(X, Y):
#         # 為 X 增加常數列
#         X = np.column_stack((np.ones(X.shape[0]), X))
        
#         # 使用公式計算係數：(X'X)^(-1)X'Y
#         beta = np.linalg.inv(X.T @ X) @ X.T @ Y

#         # 分離截距項和斜率係數
#         alpha = beta[:1]
#         coefficients = beta[1:]
#         return alpha, coefficients

#     def calculate_residuals(stock_prices, index_prices, window_length):
#         residuals = []
#         indices = []  # 用於存儲對應的索引
#         # alpha_index = stock_prices.index

#         # 使用指定的滾動窗口長度
#         for i in range(window_length, len(stock_prices)):
#             X = index_prices[i-window_length:i]
#             y = stock_prices[i-window_length:i]
#             alpha, beta = ols_regression(X.values.reshape(-1, 1), y.values)
#             alpha_series = pd.Series([alpha] * len(y), index=y.index)
#             predicted_y = X * beta
#             residual = y.iloc[-1] - predicted_y.iloc[-1] - alpha_series.iloc[-1]
#             residuals.append(residual)
#             indices.append(y.index[-1])  # 添加對應的索引
            
#         return pd.Series(residuals, index=indices)
    
#     market_ret = index_series.dropna()
#     beta_df = pd.DataFrame()
#     deleted_columns = []

#     for target in df.columns:
#         stock_ret = df[target]
#         stock_ret.index = pd.to_datetime(stock_ret.index)
#         if stock_ret.empty:
#             deleted_columns.append(target)
#         else:

#             stock_first_index = stock_ret.index[0]
#             market_first_index = market_ret.index[0]
#             stock_last_index = stock_ret.index[-1]
#             market_last_index = market_ret.index[-1]
            
#             common_start_date = max(stock_first_index, market_first_index)
#             common_end_date = min(stock_last_index, market_last_index)
#             stock_ret = stock_ret[common_start_date:common_end_date]
#             M_ret = market_ret.loc[common_start_date:common_end_date]  

#             alphas = calculate_residuals(stock_ret, M_ret, period)
#             alphas = pd.DataFrame(alphas)
#             alphas.columns = [target]
#             alpha_df = pd.concat([beta_df, alphas], axis=1)
    
#     return alpha_df


# def rolling_regression(df: pd.DataFrame, index_series: pd.Series, period: int, return_type: str = 'alpha') -> pd.DataFrame:
#     def ols_regression(X, Y):
#         X = np.column_stack((np.ones(X.shape[0]), X))
#         beta = np.linalg.inv(X.T @ X) @ X.T @ Y
#         alpha = beta[:1]
#         coefficients = beta[1:]
#         return alpha, coefficients

#     def calculate_values(stock_prices, index_prices, window_length, return_type):
#         values = []
#         indices = []

#         for i in range(window_length, len(stock_prices)):
#             X = index_prices[i-window_length:i]#.values.reshape(-1, 1)
#             y = stock_prices[i-window_length:i]#.values
#             alpha, beta = ols_regression(X.values.reshape(-1, 1), y.values)

#             if return_type == 'alpha':
#                 alpha_series = pd.Series([alpha] * len(y), index=y.index)
#                 predicted_y = X * beta
#                 residual = y.iloc[-1] - predicted_y.iloc[-1] - alpha_series.iloc[-1]
#                 values.append(residual)

#             elif return_type == 'beta':
#                 values.append(beta)
#             indices.append(y.index[-1])

#         return pd.Series(values, index=indices)
    
#     market_ret = index_series.dropna()
#     columns_df = pd.DataFrame()

#     for target in df.columns:
#         stock_ret = df[target].dropna()
#         if stock_ret.empty:
#             continue

#         stock_ret.index = pd.to_datetime(stock_ret.index)
#         market_ret.index = pd.to_datetime(market_ret.index)

#         stock_first_index = stock_ret.index[0]
#         market_first_index = market_ret.index[0]
#         stock_last_index = stock_ret.index[-1]
#         market_last_index = market_ret.index[-1]
        
#         common_start_date = max(stock_first_index, market_first_index)
#         common_end_date = min(stock_last_index, market_last_index)
#         stock_ret = stock_ret[common_start_date:common_end_date]
#         M_ret = market_ret.loc[common_start_date:common_end_date]  


#         values = calculate_values(stock_ret, M_ret, period, return_type)
#         values = pd.DataFrame(values)
#         values.columns = [target]
#         print(values)
#         columns_df = pd.concat([columns_df, values], axis=1)
#     columns_df = columns_df.applymap(lambda x: x[0] if isinstance(x, np.ndarray) else x)

#     return columns_df


def rolling_regression(regression_y: pd.DataFrame, regression_x: pd.Series | pd.DataFrame, period: int, return_type: str = 'alpha') -> pd.DataFrame:
    def ols_regression(X, Y):
        X = np.column_stack((np.ones(X.shape[0]), X))
        beta = np.linalg.inv(X.T @ X) @ X.T @ Y
        alpha = beta[:1]
        coefficients = beta[1:]
        return alpha, coefficients

    def calculate_values(regression_y, regression_x, window_length, return_type):
        values = []
        indices = []

        for i in range(window_length, len(regression_y)):
            X = regression_x[i-window_length:i]#.values.reshape(-1, 1)
            y = regression_y[i-window_length:i]#.values
            alpha, beta = ols_regression(X.values.reshape(-1, 1), y.values)

            if return_type == 'alpha':
                alpha_series = pd.Series([alpha] * len(y), index=y.index)
                predicted_y = X * beta
                residual = y.iloc[-1] - predicted_y.iloc[-1] - alpha_series.iloc[-1]
                values.append(residual)

            elif return_type == 'beta':
                values.append(beta)
            indices.append(y.index[-1])

        return pd.Series(values, index=indices)
    

    columns_df = pd.DataFrame()

    # TO DO ~~~~
    if isinstance(regression_x, pd.DataFrame):
        for target in regression_y.columns:
            if target in regression_x.columns:
                input_x = regression_x[target].dropna()
                input_y = regression_y[target].dropna()
                if input_x.empty or input_y.empty:
                    continue

                input_y.index = pd.to_datetime(input_y.index)
                input_x.index = pd.to_datetime(input_x.index)

                y_first_index = input_y.index[0]
                x_first_index = input_x.index[0]
                y_last_index = input_y.index[-1]
                x_last_index = input_x.index[-1]

                common_start_date = max(y_first_index, x_first_index )
                common_end_date = min(y_last_index, x_last_index)
                outcome_y = input_y[common_start_date:common_end_date]
                outcome_x = input_x.loc[common_start_date:common_end_date]  

                values = calculate_values(outcome_y, outcome_x, period, return_type)
                values = pd.DataFrame(values)
                values.columns = [target]
                print(values)
                columns_df = pd.concat([columns_df, values], axis=1)



    input_x = regression_x.dropna()
    for target in regression_y.columns:
        input_y = regression_y[target].dropna()
        if input_y.empty:
            continue

        input_y.index = pd.to_datetime(input_y.index)
        input_x.index = pd.to_datetime(input_x.index)

        y_first_index = input_y.index[0]
        x_first_index = input_x.index[0]
        y_last_index = input_y.index[-1]
        x_last_index = input_x.index[-1]
        
        common_start_date = max(y_first_index, x_first_index )
        common_end_date = min(y_last_index, x_last_index)
        outcome_y = input_y[common_start_date:common_end_date]
        outcome_x = input_x.loc[common_start_date:common_end_date]  


        values = calculate_values(outcome_y, outcome_x, period, return_type)
        values = pd.DataFrame(values)
        values.columns = [target]
        print(values)
        columns_df = pd.concat([columns_df, values], axis=1)
    columns_df = columns_df.applymap(lambda x: x[0] if isinstance(x, np.ndarray) else x)

    return columns_df




# def rolling_regression(df: pd.DataFrame, index_series: pd.Series | pd.DataFrame, period: int, return_type: str = 'alpha') -> pd.DataFrame:
#     def ols_regression(X, Y):
#         X = np.column_stack((np.ones(X.shape[0]), X))
#         beta = np.linalg.inv(X.T @ X) @ X.T @ Y
#         alpha = beta[0]
#         coefficients = beta[1:]
#         return alpha, coefficients

#     def calculate_values(stock_prices, index_prices, window_length, return_type):
#         values = []
#         indices = []

#         for i in range(window_length, len(stock_prices)):
#             X = index_prices[i-window_length:i]
#             y = stock_prices[i-window_length:i]
#             alpha, beta = ols_regression(X.values.reshape(-1, 1), y.values)

#             if return_type == 'alpha':
#                 alpha_series = pd.Series([alpha] * len(y), index=y.index)
#                 predicted_y = X * beta + alpha
#                 residual = y.iloc[-1] - predicted_y.iloc[-1]
#                 values.append(residual)

#             elif return_type == 'beta':
#                 values.append(beta)
#             indices.append(y.index[-1])

#         return pd.Series(values, index=indices)
    
#     columns_df = pd.DataFrame()

#     if isinstance(index_series, pd.DataFrame):
#         for target in df.columns:
#             if target in index_series.columns:
#                 market_ret = index_series[target].dropna()
#                 stock_ret = df[target].dropna()
#                 if stock_ret.empty or market_ret.empty:
#                     continue

#                 stock_ret.index = pd.to_datetime(stock_ret.index)
#                 market_ret.index = pd.to_datetime(market_ret.index)

#                 common_start_date = max(stock_ret.index[0], market_ret.index[0])
#                 common_end_date = min(stock_ret.index[-1], market_ret.index[-1])

#                 stock_ret = stock_ret[common_start_date:common_end_date]
#                 market_ret = market_ret[common_start_date:common_end_date]

#                 values = calculate_values(stock_ret, market_ret, period, return_type)
#                 columns_df[target] = values
#     else:
#         market_ret = index_series.dropna()
#         for target in df.columns:
#             stock_ret = df[target].dropna()
#             if stock_ret.empty:
#                 continue

#             stock_ret.index = pd.to_datetime(stock_ret.index)
#             market_ret.index = pd.to_datetime(market_ret.index)

#             common_start_date = max(stock_ret.index[0], market_ret.index[0])
#             common_end_date = min(stock_ret.index[-1], market_ret.index[-1])

#             stock_ret = stock_ret[common_start_date:common_end_date]
#             market_ret = market_ret[common_start_date:common_end_date]

#             values = calculate_values(stock_ret, market_ret, period, return_type)
#             columns_df[target] = values

#     return columns_df