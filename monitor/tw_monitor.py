import os
import json
import pandas as pd
from IPython.display import display
import tejapi
from datetime import date
from dateutil.relativedelta import relativedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from data_center.tej_handler import TEJHandler

class TWMarketMonitor(TEJHandler):
    def __init__(self):
        super().__init__()
        self.tw_monitor_config = self.get_tw_monitor_config()  
        self.client_url = self.tw_monitor_config['mongodb']['client_url']

    # ==================== data center config ==================== #

    def get_tw_monitor_config(self):
        with open(f'{PROJECT_ROOT}/monitor/config/tw_moniter.json') as f:
            tw_monitor_config = json.loads(f.read())
        return tw_monitor_config


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
    
    def get_revenue_returns(self, review_month: int=1):
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
    



