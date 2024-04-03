import os
import json
import pandas as pd
from IPython.display import display
import tejapi
from datetime import date
from dateutil.relativedelta import relativedelta

from pymongo import MongoClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from data_center.tej_handler import TEJHandler

class TWMarketMonitor(TEJHandler):
    def __init__(self):
        super().__init__()
        self.tw_monitor_config = self.get_tw_monitor_config()  
        self.client_url = self.tw_monitor_config['mongodb']['client_url']
        self.database_list = self.tw_monitor_config['mongodb']['database']
        self.colletion_list = self.tw_monitor_config['mongodb']['collection']

    # ==================== data center config ==================== #

    def get_tw_monitor_config(self):
        with open(f'{PROJECT_ROOT}/monitor/config/tw_moniter.json') as f:
            tw_monitor_config = json.loads(f.read())
        return tw_monitor_config


    # ========================= get Pstage data from MongoDB ===================== #

    def get_data_from_mongo(self, database_str:str = 'PSTAGE', collection_str:str = 'financial_report', limit = 10000):
        mongo = MongoClient(self.client_url)
        collection = mongo[database_str][collection_str]
        
        if collection_str not in self.colletion_list:
            raise ValueError(f"please use one of {self.colletion_list} in collection_str")
        
        # 根据集合名决定排序字段
        if collection_str == "financial_report":
            sort_target = '財務資料日'
        elif collection_str in ["margin_trading", "three_major_investors_activity", "securities_trading_data", "securities_returns"]:
            sort_target = '日期'
        elif collection_str == "monthly_revenue":
            sort_target = '資料日期'
        elif collection_str in ["tsx_property", "securities_property"]:     
            sort_target = '目前狀態'
        
        df = pd.DataFrame(list(collection.find().sort(sort_target, -1).limit(limit)))
        return df

    # ========================= get TEJ API information ===================== #

    def get_market_returns(self, review_month: int=1):
        today = date.today()
        last_month_date = today - relativedelta(months=review_month)
        last_month_date_str = last_month_date.strftime('%Y-%m-%d')

        source_code = "TWN/EWPRCD2"
        raw_data = tejapi.get(source_code, 
                              chinese_column_name=True, 
                              mdate={'gt':last_month_date_str},
                              paginate=True)
        pdata_df = TEJHandler().pdata_pipline(raw_data, source_code)
        pmart = TEJHandler().pmart_pipline(pdata_df, source_code)

        keys_list = list(pmart.keys())
        resent_returns_df = pmart[keys_list[0]]
        return resent_returns_df
    
    # def get_revenue_returns(self, review_month: int=1):
    #     today = date.today()
    #     last_month_date = today - relativedelta(months=review_month)
    #     last_month_date_str = last_month_date.strftime('%Y-%m-%d')

    #     source_code = "TWN/EWPRCD2"
    #     raw_data = tejapi.get(source_code, 
    #                           chinese_column_name=True, 
    #                           mdate={'gt':last_month_date_str},
    #                           paginate=True)
    #     pdata_df = TEJHandler().pdata_pipline(raw_data, source_code)
    #     pmart = TEJHandler().pmart_pipline(pdata_df, source_code)

    #     keys_list = list(pmart.keys())
    #     resent_returns_df = pmart[keys_list[0]]
    #     return resent_returns_df
    



