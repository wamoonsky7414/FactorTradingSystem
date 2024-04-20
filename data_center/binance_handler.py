import os
import json
import pandas as pd
import time

import asyncio
import requests
import pytz
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class BinanceHandHandler(object):
    def __init__(self,select_symbol=None):

        self.data_center_config = self.get_data_center_config()

        self.base_url = self.data_center_config["basic_url"]
        self.endpoint_list = self.data_center_config["endpoint_list"]
        self.default_symbol = self.data_center_config['default_symbol']
        self.select_symbol = select_symbol or self.default_symbol
        self.timezone = self.data_center_config["timezone"]
        self.contracttype = self.data_center_config['contracttype']
        self.interval = self.data_center_config['interval']
        self.start_time = self.data_center_config['since']


    # ==================== data center config ==================== #
    def get_data_center_config(self):
        with open(f'{PROJECT_ROOT}/data_center/config/binance_handler_config.json') as f:
            data_center_config = json.loads(f.read())
        return data_center_config
    
    # def get_data_center_config(self):
    #     with open(f'{PROJECT_ROOT}/data_center/config/binance_handler_config_total_market.json') as f:
    #         data_center_config = json.loads(f.read())
    #     return data_center_config
    # binance_handler_config_total_market
    
    # ==================== get data ======================== #
    def get_origin_data(self, target:str):
        if self.contracttype == "PERPETUAL":
            contracttype_file = 'UPERP'
        file_path = rf'{PROJECT_ROOT}/data/CRYPTO/BINANCE/ORIGIN/{contracttype_file}/ohlcv/{self.interval}/{target}.csv'
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return df
    
    def get_factor_data(self, factor:str):
        if self.contracttype  == "PERPETUAL":
            contracttype_file = 'UPERP'
        file_path = rf'{PROJECT_ROOT}/data/CRYPTO/BINANCE/FACTOR/{contracttype_file}/{self.interval}/{factor}.csv'
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)#.dropna()
        return df

    
    # ===================== arrange original to factor =============== #
    def arrange_data_to_ohlcv_factor(self):
        df_list = []
        for symbol in self.select_symbol:
            df = BinanceHandHandler().get_origin_data(symbol)
            df_list.append(df)
        factor_ohlcv = df_list[0].columns.tolist()
        if self.contracttype  == "PERPETUAL":
            contracttype_file = 'UPERP'
        folder_path = f'{PROJECT_ROOT}/data/CRYPTO/BINANCE/FACTOR/{contracttype_file}/{self.interval}'
        for factor in factor_ohlcv:
            for i, symbol in enumerate(self.select_symbol):
                tmp_df = df_list[i]
                tmp_df = tmp_df[[factor]]
                tmp_df.columns = [symbol]
                if i == 0:
                    factor_df = tmp_df
                else:
                    factor_df = pd.concat([factor_df, tmp_df], axis=1)

            file_path = rf'{folder_path}/{factor}.csv'
            factor_df.to_csv(file_path, index=True)
        return 'Finish'
    
    # ======================= update binance ohlcv data ===================== #

    def update_ohlcv_data_from_binance(self):

        # target_timezone = pytz.timezone(self.timezone)

        if self.contracttype == "PERPETUAL":
            contracttype_file = 'UPERP'

            '''
            This code can be deleted after build the other interval
            '''
            for target in self.select_symbol:

                yesterday = datetime.now() - timedelta(days=1)
                yesterday_timestamp_ms = int(yesterday.timestamp() * 1000)
                end_time = yesterday_timestamp_ms

                file_path = rf'{PROJECT_ROOT}/data/CRYPTO/BINANCE/ORIGIN/{contracttype_file}/ohlcv/{self.interval}/{target}.csv'
                try:
                    df = pd.read_csv(file_path)
                    last_time = pd.to_datetime(df.iloc[-1]['datetime'])
                    print(f"Loading: {target}, Last record time: {last_time}, {self.timezone}")  
                    next_day = last_time + pd.Timedelta(days=1)
                    start_time = int(pd.Timestamp(next_day, tz=self.timezone).timestamp() * 1000)

                except FileNotFoundError:
                    print(f'Generateing new file for {target} and loading data from Binance' )
                    df = pd.DataFrame(columns=['datetime','open','high','low','close','volume', 'volvalue', 'takerbuy', 'takerbuyvalue']) 
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    df.to_csv(file_path, index=False)
                    start_time = int(pd.Timestamp(self.start_time, tz=self.timezone).timestamp() * 1000)

                pair = target
                ContractType = self.contracttype
                interval = self.interval
                start_time = start_time
                end_time = end_time
                limit = 1000



                # Make requests in chunks until you reach the end_time
                while start_time < end_time:
                    params = {
                        "pair": pair,
                        "ContractType": ContractType,
                        "interval": interval,
                        "startTime": start_time,
                        "endTime": end_time,
                        "limit": limit  # Use the chunk_size for each request
                    }

                    url = self.base_url + self.endpoint_list["continuousklines"]
                    response = requests.get(url, params=params)
                    data = response.json()
                    
                    # If data is empty, break out of the loop
                    if not data:
                        break

                    new_df = pd.DataFrame({
                        'datetime': [row[0] for row in data],
                        'open': [row[1] for row in data],
                        'high': [row[2] for row in data],
                        'low': [row[3] for row in data],
                        'close': [row[4] for row in data],
                        'volume': [row[5] for row in data],
                        'volvalue': [row[7] for row in data],
                        'takerbuy': [row[8] for row in data],
                        'takerbuyvalue': [row[9] for row in data],
                    })

                    # Convert 'datetime' from timestamp (milliseconds) to datetime format
                    new_df['datetime'] = pd.to_datetime(new_df['datetime'], unit='ms')

                    # If you want to ensure the datetime is formatted as a string in the specific format "YYYY-MM-DD HH:MM:SS"
                    new_df['datetime'] = new_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    df = pd.concat([df, new_df],ignore_index=True)
                    df.to_csv(file_path, index=False)

                    start_time = int(data[-1][0]) + 1  # Set the new start_time to the next timestamp
                    time.sleep(1)
            print(rf'Finish' '\n')

    def update_binanace_ohlcv_data_and_arrange_it_to_become_factor(self):
        BinanceHandHandler().update_ohlcv_data_from_binance()
        BinanceHandHandler().arrange_data_to_ohlcv_factor()
        return 'Finish'

    # ======================= update binance fundingrate data ===================== #

    def update_fundingrate_data_from_binance(self):
        url = self.base_url + self.endpoint_list["fundingrate"]

        yesterday = datetime.now() - timedelta(days=1)
        yesterday_timestamp_ms = int(yesterday.timestamp() * 1000)
        end_time = yesterday_timestamp_ms

        for target in self.select_symbol:

            file_path = rf'{PROJECT_ROOT}/data/CRYPTO/BINANCE/ORIGIN/UPERP/funding_rate/{target}.csv'

            try:
                df = pd.read_csv(file_path)
                last_time = pd.to_datetime(df.iloc[-1]['fundingTime'])
                print(f"Loading: {target}, Last record time: {last_time}, {self.timezone}")  
                next_day = last_time + pd.Timedelta(days=1)
                start_time = int(pd.Timestamp(next_day, tz=self.timezone).timestamp() * 1000)

            except FileNotFoundError:
                print(f'Generateing new file for {target} and loading data from Binance' )
                df = pd.DataFrame(columns=['fundingTime', 'fundingRate'])
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                df.to_csv(file_path, index=False)
                start_time = int(pd.Timestamp(self.start_time, tz=self.timezone).timestamp() * 1000)

            while start_time < end_time:
                url = self.base_url + self.endpoint_list["fundingrate"]
                response = requests.get(url, params={
                    "symbol": target,
                    "startTime": start_time,
                    "endTime": end_time,
                    "limit" : 1000,
                })

                if response.status_code == 200:
                    data = response.json()

                    if not data:
                        break

                    new_df = pd.DataFrame(data)

                    new_df['fundingTime'] = pd.to_datetime(new_df['fundingTime'], unit='ms')#, utc=True)
                    # its timezone is Asia even I use utc = True ...

                    new_df['fundingTime'] = new_df['fundingTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    new_df = new_df.drop(['symbol', 'markPrice'],axis =1)
                    # The columns "markPrice" = Close as your timezone

                    df = pd.concat([df, new_df],ignore_index=True)
                    df.to_csv(file_path, index=False)

                    start_time = int(data[-1]['fundingTime']) + 1
                    time.sleep(1)

                else:
                    print("Failed to fetch data: Status code", response.status_code)

    # def file_explorer(self):
    #     if self.contracttype == "PERPETUAL":
    #         contracttype_file = 'UPERP'
    #     elif self.contracttype:
    #         contracttype_file = 'SOPT'
    #         #GET /api/v3/klines
    #     elif:


