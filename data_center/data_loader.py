import requests
import pytz
import pandas as pd

def Loaddata(target_list:list):
    for target in target_list:
        target_timezone = pytz.timezone("Asia/Hong_Kong")
        print(target_timezone)

        pair = target
        ContractType = "PERPETUAL"
        interval = "8h"
        start_time = int(pd.Timestamp("2020-01-01", tz=target_timezone).timestamp() * 1000) #long
        end_time = int(datetime.now().timestamp() * 1000) #long
        limit = 1000

        save_path = rf'/Users/tedting/Documents/Blockchain/binance_data/UPERP/8H'
        file_name = rf'{target}_8h.csv'

        all_data = []

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
            all_data.extend(data)
            # Create a DataFrame from the combined data
            df = pd.DataFrame({
                'datetime': [row[0] for row in all_data],
                'open': [row[1] for row in all_data],
                'high': [row[2] for row in all_data],
                'low': [row[3] for row in all_data],
                'close': [row[4] for row in all_data],
                'volume': [row[5] for row in all_data]
            })

            # Convert 'datetime' from timestamp (milliseconds) to datetime format
            df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')

            # If you want to ensure the datetime is formatted as a string in the specific format "YYYY-MM-DD HH:MM:SS"
            df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

            df.to_csv(f'{save_path}/{file_name}', index=False)

            # Update the start_time for the next request
            start_time = int(data[-1][0]) + 1  # Set the new start_time to the next timestamp
            time.sleep(1)
        print('Finish!')