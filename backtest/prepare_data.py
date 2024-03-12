import pandas as pd
import numpy as np
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, PROJECT_ROOT) 

class PrepareData(object):
    def __init__(self, market, data_path):
        self.market = market
        self.data_path = data_path


    # ================ get factor data ==================== #
    def get_factor_df(self, data_path):
        if self.market == 'CRYPTO':
           factor = get_crpyto_factor_df()
           factor = pd.read_csv(f'{data_path}')

        else:    
            factor = pd.read_csv(f'{data_path}')
        return factor
    
    def get_projectroot():
        return PROJECT_ROOT


def get_crpyto_factor_df(self):
    df = pd.DataFrame()

    pd.read_csv(f'{data_path}')

def data_reshape(symbol_list:list, column:str, file_path_template:str):
    df = pd.DataFrame()
    for symbol in symbol_list:
        file_path = f'{file_path_template}{symbol}.csv'
        tmp_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        tmp_df = tmp_df[[column]]
        tmp_df.columns = [symbol]
        df = pd.concat([df, tmp_df], axis=1).dropna()
    return df


