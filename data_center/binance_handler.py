import os
import json
import pandas as pd

import asyncio
import requests
import pytz
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class BinanceHandHandler(object):
    def __init__(self,select_symbol=None):

        self.data_center_config = self.get_data_center_config()

        self.default_symbol = self.data_center_config['default_symbol']
        self.select_symbol = select_symbol or self.default_symbol
        self.contracttype = self.data_center_config['contracttype']
        self.interval = self.data_center_config['interval']


    # ==================== data center config  ==================== #
    def get_data_center_config(self):
        with open(f'{PROJECT_ROOT}/data_center/config/data_center_config.json') as f:
            data_center_config = json.loads(f.read())
        return data_center_config
    
    def update_data_from_binance(self):
        if self.contracttype == "PERPETUAL":
            contracttype_file = 'UPERP'
        for target in self.select_symbol:
            print(target)
            file_path = rf'{PROJECT_ROOT}/data/CRYPTO/BINANCE/{contracttype_file}/{self.interval}/{target}.csv'
            try:
                df = pd.read_csv(file_path)

                '''
                This code need to be contribute after finish the update_loader

                last_time = pd.to_datetime(df.iloc[-1]['timestamp'])
                print(f"Last record time: {last_time}")
                
                '''

            except FileNotFoundError:
                df = pd.DataFrame(columns=['datetime','open','high','low','close','volume']) 
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                df.to_csv(file_path, index=False)
                

            target_timezone = pytz.timezone("Asia/Hong_Kong")
            print(target_timezone)

            pair = target
            ContractType = self.contracttype
            interval = self.interval
            start_time = int(pd.Timestamp("2016-01-01 00:00:00", tz=target_timezone).timestamp() * 1000) 
            end_time = int(datetime.now().timestamp() * 1000) 
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
                
                url = "https://fapi.binance.com/fapi/v1/continuousKlines"
                response = requests.get(url, params=params)
                data = response.json()
                
                # If data is empty, break out of the loop
                if not data:
                    break
                
                # Append the data to the all_data list
                df['datetime'] = [row[0] for row in data]
                df['open'] = [row[1] for row in data]
                df['high'] = [row[2] for row in data]
                df['low'] = [row[3] for row in data]
                df['close'] = [row[4] for row in data]
                df['volume'] = [row[5] for row in data]
                
                print(df)
                # Create a DataFrame from the combined data
                # df = pd.DataFrame({
                #     'datetime': [row[0] for row in all_data],
                #     'open': [row[1] for row in all_data],
                #     'high': [row[2] for row in all_data],
                #     'low': [row[3] for row in all_data],
                #     'close': [row[4] for row in all_data],
                #     'volume': [row[5] for row in all_data]
                # })

                # Convert 'datetime' from timestamp (milliseconds) to datetime format
                df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')

                # If you want to ensure the datetime is formatted as a string in the specific format "YYYY-MM-DD HH:MM:SS"
                df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
                print(df)
                df.to_csv(file_path, index=False)

                # Update the start_time for the next request
                start_time = int(data[-1][0]) + 1  # Set the new start_time to the next timestamp
                time.sleep(1)
            print('Finish!')