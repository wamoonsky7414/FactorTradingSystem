import os
import json
import pandas as pd
from IPython.display import display
import tejapi

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TEJHandler(object):
    def __init__(self):
        self.data_center_config = self.get_data_center_config()

        self.api_key = self.data_center_config['api_key']
        self.ignoretz = self.data_center_config['ignoretz']
        self.tej_data_table = self.data_center_config['tej_data_table']

        tejapi.ApiConfig.api_key = self.api_key
        tejapi.ApiConfig.ignoretz = self.ignoretz

    # ==================== data center config ==================== #

    def get_data_center_config(self):
        with open(f'{PROJECT_ROOT}/data_center/config/tej_handler_config.json') as f:
            data_center_config = json.loads(f.read())
        return data_center_config

    # ======================== get tej permission inf ================ #
        
    def get_info(self):
        info = tejapi.ApiConfig.info()
        print(info)

    def get_api_key(self):
        print(f"API KEY is '{self.api_key}'")
    
    def get_tej_data_table(self):
        df = pd.DataFrame([self.tej_data_table]).T
        display(df)

    # ======================== arrange raw data to Pdata ================ #

    def pdata_pipline(self, df: pd.DataFrame, data_code: str):
        df = self.rename_symbol_and_datetime(df, data_code)
        df_dict = self.unstack_data(df, data_code)
        return df_dict 

    def rename_symbol_and_datetime(self, df: pd.DataFrame, data_code: str):
        if data_code in ["TWN/EWGIN", "TWN/EWPRCD", "TWN/EWPRCD2", "TWN/EWTINST1"]:
            df.rename(columns={'證券碼': 'symbol', '日期': 'datetime'}, inplace=True)
        elif data_code == "TWN/EWSALE":
            df.rename(columns={'證券碼': 'symbol', '資料日期': 'datetime'}, inplace=True)
        elif data_code == "TWN/EWIFINQ":
            df.rename(columns={'證券碼': 'symbol', '財務資料日': 'datetime'}, inplace=True)
        elif data_code == "TWN/EWIPRCD":
            df.rename(columns={'指數碼': 'symbol', '日期': 'datetime'}, inplace=True)
        elif data_code == "TWN/EWIPRCSTD":
            df.rename(columns={'指數碼': 'symbol', '目前狀態': 'datetime'}, inplace=True)
        elif data_code == "TWN/EWNPRCSTD":
            df.rename(columns={'證券碼': 'symbol', '目前狀態': 'datetime'}, inplace=True)
        else:
            print("Unsupported data code. Please use one of the following data codes with the correct column names to rename:")
            print(self.tej_data_table)
        return df

    def unstack_data(self, df: pd.DataFrame, data_code: str):
        df_dic = {}
        column_list = df.columns.to_list()
        for column in column_list[2:]:
            unstack_df = df.reset_index().set_index(['datetime', 'symbol'])[column].unstack()
            df_dic[column] = unstack_df
        return df_dic
    
    # =========================== new function after 2024/03/27 ====================== #

    def get_daily_frequent(self, source_dict: dict, data_code: str, signal_delay: int= None):
        
        factor_dict = {}

        date_df_code = "TWN/EWIPRCD"
        date_df = tejapi.get(date_df_code, chinese_column_name=True)
        for_date_dict = self.pdata_pipline(date_df, date_df_code)

        if data_code == "TWN/EWSALE":
            release = '營收發布日'
        elif data_code == "TWN/EWIFINQ":
            release = '財報發布日'
        elif data_code in ["TWN/EWIPRCSTD", "TWN/EWNPRCSTD"]:
            for column in list(source_dict.keys()):
                data_reindex = source_dict[column].reindex(for_date_dict['指數收盤價'].index, method = 'ffill')
                factor_dict[column] = data_reindex 
            return factor_dict
        
        for column in list(source_dict.keys()):
            releaserankpct = source_dict[release].rank(axis=1,pct=True)
            filter = source_dict[column][releaserankpct < signal_delay]
            filter.index = source_dict[release][releaserankpct < signal_delay].max(axis=1)
            data_reindex = filter.reindex(for_date_dict['指數收盤價'].index, method = 'ffill')
            factor_dict[column] = data_reindex 
        return factor_dict

    def pmart_pipline(self, source_dict: dict, data_code: str, signal_delay: int= None):

        if type(signal_delay) == int:
            signal_delay *=  0.01    

        if data_code in ["TWN/EWGIN", "TWN/EWPRCD", "TWN/EWPRCD2", "TWN/EWTINST1", "TWN/EWIPRCD"]:
            factor_dict = source_dict
        elif data_code in ["TWN/EWSALE", "TWN/EWIFINQ", "TWN/EWIPRCSTD", "TWN/EWNPRCSTD"]:
            factor_dict = self.get_daily_frequent(source_dict, data_code, signal_delay)
        else:
            print("Unsupported data code. Please use one of the following data codes with the correct column names to rename:")
            print(self.tej_data_table)
        return factor_dict
