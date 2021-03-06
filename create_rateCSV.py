import ast
import json
import requests
import pandas as pd
import os
import multiprocessing


months = list(range(1, 13)) #月
days_default = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] #日数
days_leap = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] #うるう年

#時、分のリスト。計算量的に先にやったほうがよさそう。
hours = [str(hour).zfill(2) for hour in range(24)]
minutes = [str(minute*5).zfill(2) for minute in range(12)]



class create_rateCSV:

    #年とコインの種類。btc_jpyなど。
    def __init__(self, year, coin):
        self.year = year
        self.coin = coin

        if self.year%4==0:
            self.days = days_leap
        else:
            self.days = days_default
    
    #めちゃくちゃ時間かかりそうなので、サブプロセスでやるために、何月かを受け取って処理する。外部から、mapで回すということ。
    #本番環境では外部ファイルのmainで運用するので、同じように適用可能
    def create_CSV(self, month):
        
        os.chdir('d:/workSpace/virtual_coin/')
        if not os.path.exists(self.coin+'/'+str(self.year)):
            os.makedirs(self.coin+'/'+str(self.year))
        
        csv_dir = str(self.coin)+'/'+str(self.year)

        print('{}月を実行中'.format(month))
        #月ごとにしないとメモリ死にそうだったので...。
        df_rates = pd.DataFrame(columns=['utc_timestamp', 'rate'])
        csv_path = csv_dir+'/'+str(month)+'_month.csv'
        
        if os.path.exists(csv_path):
            print('{}月データは既に存在しています'.format(month))
            return

        f=open(csv_path, mode='w')
        f.close()

        for day in range(1, self.days[month-1]+1):
            
            #以下時刻をループで回す
            for hour in hours:
                for minute in minutes:
                    moment = 'T'+hour+':'+minute+':00:000Z'
                    time =  str(self.year)+'-'+str(month).zfill(2)+'-'+str(day).zfill(2)+moment

                    rate_data = requests.get('https://coincheck.com/ja/exchange/rates/search', params={'pair':self.coin, 'time':time})
                    rate_dict = ast.literal_eval(rate_data.text)
                    rate = float(rate_dict['rate'])

                    df_rates = df_rates.append({'utc_timestamp':time,'rate':rate}, ignore_index=True)
            

        df_rates.to_csv(csv_path, index=False)
        print('{}月終わり'.format(month))

                        


    
    #現在の2021年（もうすぐ2022年）について、現在までのレートを埋めて更新する。
    def update_csv(self):
        return 0
    
    



    


            

        