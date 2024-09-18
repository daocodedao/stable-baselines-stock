import baostock as bs
import pandas as pd
import os


OUTPUT = './stockdata'


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class Downloader(object):
    def __init__(self,
                 output_dir,
                 date_start='1990-01-01',
                 date_end='2020-03-23',
                 stock_code = None):
        self._bs = bs
        bs.login()
        self.date_start = date_start
        # self.date_end = datetime.datetime.now().strftime("%Y-%m-%d")
        self.date_end = date_end
        self.output_dir = output_dir
        self.fields = "date,code,open,high,low,close,volume,amount," \
                      "adjustflag,turn,tradestatus,pctChg,peTTM," \
                      "pbMRQ,psTTM,pcfNcfTTM,isST"
        self.stock_code = stock_code

    def exit(self):
        bs.logout()

    def get_codes_by_date(self, date):
        print(date)
        stock_rs = bs.query_all_stock(date)
        stock_df = stock_rs.get_data()
        print(stock_df)
        return stock_df

    def run(self):
        if not self.stock_code:
            stock_df = self.get_codes_by_date(self.date_end)
            for index, row in stock_df.iterrows():
                print(f'processing {row["code"]} {row["code_name"]}')
                df_code = bs.query_history_k_data_plus(row["code"], self.fields,
                                                    start_date=self.date_start,
                                                    end_date=self.date_end).get_data()
                df_code.to_csv(f'{self.output_dir}/{row["code"]}.{row["code_name"]}.csv', index=False)
        else:
            
            stockData = bs.query_stock_basic(self.stock_code).get_data()
            stockName = stockData['code_name'].values[0]
            print(f'processing {self.stock_code} {stockName}')
            df_code = bs.query_history_k_data_plus(self.stock_code, self.fields,
                                                    start_date=self.date_start,
                                                    end_date=self.date_end).get_data()
            df_code.to_csv(f'{self.output_dir}/{stockData}.{stockName}.csv', index=False)
            
        self.exit()


if __name__ == '__main__':
    # 获取全部股票的日K线数据
    mkdir('./stockdata/train')
    downloader = Downloader('./stockdata/train', 
                            date_start='1990-01-01', 
                            date_end='2019-11-30',
                            stock_code="sh.600755")
    downloader.run()

    mkdir('./stockdata/test')
    downloader = Downloader('./stockdata/test', 
                            date_start='2019-12-01', 
                            date_end='2024-09-13',
                            stock_code="sh.600755")
    downloader.run()

