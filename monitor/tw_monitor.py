import os
import json
import pandas as pd
from IPython.display import display
import tejapi
from datetime import datetime
import bson
from datetime import date
from dateutil.relativedelta import relativedelta

from pymongo import MongoClient

import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1,PROJECT_ROOT)

from data_center.tej_handler import TEJHandler


class TWMarketMonitor(TEJHandler):
    def __init__(self):
        super().__init__()
        self.tw_monitor_config = self.get_tw_monitor_config()  
        self.client_url = self.tw_monitor_config['mongodb']['client_url']
        self.database_list = self.tw_monitor_config['mongodb']['database']
        self.colletion_list = self.tw_monitor_config['mongodb']['collection']
        self.limit_for_company_num = self.tw_monitor_config['limit_for_company_num']


    # ==================== data center config ==================== #

    def get_tw_monitor_config(self):
        with open(f'{PROJECT_ROOT}/monitor/config/tw_moniter.json') as f:
            tw_monitor_config = json.loads(f.read())
        return tw_monitor_config


    # ========================= get Pstage data from MongoDB ===================== #

    def get_data_from_mongo(self, database_str:str = 'PSTAGE', collection_str:str = 'financial_report', start_date="2020-01-01"):
        # if limit is None:
        #     limit = self.limit_for_company_num
        mongo = MongoClient(self.client_url)
        collection = mongo[database_str][collection_str]
        
        if collection_str not in self.colletion_list:
            raise ValueError(f"please use one of {self.colletion_list} in collection_str")
        
        if collection_str == "financial_report":
            date_field = '財務資料日'
        elif collection_str in ["margin_trading", "three_major_investors_activity", "securities_trading_data", "securities_returns", "tsx"]:
            date_field = '日期'
        elif collection_str == "monthly_revenue":
            date_field = '資料日期'
        elif collection_str in ["tsx_property", "securities_property"]:     
            date_field = '目前狀態'

        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        start_date_bson = bson.datetime.datetime(start_date.year, start_date.month, start_date.day)

        # Build the query using the converted start_date
        query = {date_field: {"$gt": start_date_bson}}

        df = pd.DataFrame(list(collection.find(query).sort(date_field, -1)))        
        # df = pd.DataFrame(list(collection.find(query).sort(date_field, -1).limit(limit)))
        return df

    
    # ======================== arrange raw data to Pdata ================ #

    def pdata_pipline(self, df: pd.DataFrame, collection_str:str = 'financial_report'):
        df = self.rename_symbol_and_datetime(df, collection_str)
        df_dict = self.unstack_data(df)
        return df_dict 

    def rename_symbol_and_datetime(self, df: pd.DataFrame, collection_str: str):
        if collection_str not in self.colletion_list:
            raise ValueError(f"please use one of {self.colletion_list} in collection_str")
        
        if collection_str in ["margin_trading", "securities_trading_data", "securities_returns", "three_major_investors_activity"]:
            df.rename(columns={'證券碼': 'symbol', '日期': 'datetime'}, inplace=True)
        elif collection_str == "monthly_revenue":
            df.rename(columns={'證券碼': 'symbol', '資料日期': 'datetime'}, inplace=True)
        elif collection_str == "financial_report":
            df.rename(columns={'證券碼': 'symbol', '財務資料日': 'datetime'}, inplace=True)
        elif collection_str == "tsx":
            df.rename(columns={'指數碼': 'symbol', '日期': 'datetime'}, inplace=True)
        elif collection_str == "tsx_property":
            df.rename(columns={'指數碼': 'symbol', '目前狀態': 'datetime'}, inplace=True)
        elif collection_str == "securities_property":
            df.rename(columns={'證券碼': 'symbol', '目前狀態': 'datetime'}, inplace=True)
  
        return df

    def unstack_data(self, df: pd.DataFrame):
        df_dic = {}
        column_list = df.columns.to_list()
        for column in column_list[3:]:
            cleaned_df = df.drop_duplicates(subset=['datetime', 'symbol'], keep='first') # be careful this function will effect other collection or not
            unstack_df = cleaned_df.reset_index().set_index(['datetime', 'symbol'])[column].unstack()
            df_dic[column] = unstack_df
        return df_dic

    def get_daily_frequent(self, source_dict: dict, collection_str: str, signal_delay= 0.9, start_date="2020-01-01"):
        # if limit is None:
        #     limit = self.limit_for_company_num
        
        factor_dict = {}

        tsx_pstage = self.get_data_from_mongo(database_str= 'PSTAGE', collection_str = 'tsx' , start_date = start_date)
        for_date_dict = self.pdata_pipline(tsx_pstage, 'tsx')

        if collection_str == "monthly_revenue":
            release = '營收發布日'
        elif collection_str == "financial_report":
            release = '財報發布日'
        elif collection_str in ["tsx_property", "securities_property"]:
            for column in list(source_dict.keys()):
                data_reindex = source_dict[column].reindex(for_date_dict['指數收盤價'].index, method = 'ffill')
                factor_dict[column] = data_reindex 
            return factor_dict
        
        for column in list(source_dict.keys()):
            releaserankpct = source_dict[release].rank(axis=1,pct=True)
            df_filter = source_dict[column][releaserankpct < signal_delay]
            df_filter.index = source_dict[release][releaserankpct < signal_delay].max(axis=1)
            df_filter = df_filter[df_filter.index.isna()==False].sort_index()
            data_reindex = df_filter.reindex(for_date_dict['指數收盤價'].index, method = 'ffill')
            factor_dict[column] = data_reindex 
        return factor_dict

    def pmart_pipline(self, source_dict: dict, collection_str: str, signal_delay= 0.9 , start_date="2020-01-01"):
        if collection_str not in self.colletion_list:
            raise ValueError(f"please use one of {self.colletion_list} in collection_str")  

        if collection_str in ["margin_trading", "securities_trading_data", "securities_returns", "three_major_investors_activity", "tsx"]:
            factor_dict = source_dict
        elif collection_str in ["monthly_revenue"]:
            factor_dict = self.get_daily_frequent(source_dict, collection_str, signal_delay, start_date)
        elif collection_str in ["financial_report"]:
            limit = len(source_dict[list(source_dict)[0]].columns)*90
            factor_dict = self.get_daily_frequent(source_dict, collection_str, signal_delay, start_date)
        elif collection_str in ["tsx_property", "securities_property"]:
            factor_dict = source_dict

        return factor_dict
    
    def directly_get_pmart(self, collection, signal_delay = 0.9, start_date="2020-01-01"):

        pstage = self.get_data_from_mongo('PSTAGE', collection, start_date)
        pdata = self.pdata_pipline(pstage, collection)
        pmart = self.pmart_pipline(pdata, collection, signal_delay)
        return pmart
    

    # ======================== get monitor need info ================ #

    def get_position_open(self, ticker_list: list) -> pd.DataFrame:
        ohlcv_dic = self.directly_get_pmart('securities_trading_data', start_date="2020-01-01")
        recent_open = ohlcv_dic['開盤價']
        return recent_open.loc[:, ticker_list]




