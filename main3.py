
# https://github.com/chijunping/pythondict-quant
import os
import sys
import traceback

import pandas as pd
import random
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from rlenv.StockTradingEnv0 import StockTradingEnv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from concurrent import futures
import utils.mysql_utils as mysqlUtils

font = fm.FontProperties(fname='font/wqy-microhei.ttc')
# plt.rc('font', family='Source Han Sans CN')
plt.rcParams['axes.unicode_minus'] = False
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

host = "127.0.0.1"
port = 3306
user = "root"
password = '123456'
database = 'zeus'
columns = ['open', 'high', 'low', 'close', 'volume', 'turn',
           'sar', "ma5", "ma10", "ma20", "k", "d", "j", "talib_diff", "talib_dea",
           "talib_macd"
           ]


def modelTrain(stock_code, start_date, end_date, model_path, reTrain=True):
    """
    当模型不存在，或者显示要求重建模型时，会更新模型，否则使用已存在的模型
    :param stock_file:
    :param model_path:
    :param reTrain:
    :return:
    """
    if os.path.exists(model_path) and (not reTrain):
        return
    print(f"训练模型，当前股票：{stock_code}")
    conn = mysqlUtils.connect_database(host, port, user, password, database)
    sqlStr = f"select * from quant_stock_data where `code`='{stock_code}' and `date`>='{start_date}' and `date`<='{end_date}' order by `date`"
    df = mysqlUtils.read_data(dbInstance=conn, sql=sqlStr)
    # df['date'] = [datetime.datetime.strptime(i, "%Y%m%d") for i in df['date']]
    # 国内市场不需要此字段，所以赋值为0即可
    df = df[columns]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    # data.dropna(inplace=True)
    # data["ps_ttm"] = data["ps_ttm"].fillna(0)
    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: StockTradingEnv(df)])
    # model = PPO2(MlpPolicy, env, verbose=0, tensorboard_log='./log', seed=111, n_cpu_tf_sess=1)
    # model = PPO("MlpPolicy", env=env, tensorboard_log='./log', seed=111)
    model = PPO("MlpPolicy", env=env, seed=111)
    model.learn(total_timesteps=10000)
    model.save(model_path)


def modelTest(stock_code, start_date, end_date, model_path):
    """
    模型预测
    :param stock_file_test:
    :param model_path:
    :return:
    """
    day_profits = []
    conn = mysqlUtils.connect_database(host, port, user, password, database)
    sqlStr = f"select * from quant_stock_data where `code`='{stock_code}' and `date`>='{start_date}' and `date`<='{end_date}' order by `date`"
    df_test = mysqlUtils.read_data(dbInstance=conn, sql=sqlStr)
    # df['date'] = [datetime.datetime.strptime(i, "%Y%m%d") for i in df['date']]
    # 国内市场不需要此字段，所以赋值为0即可
    df_test = df_test[columns]
    df_test.dropna(inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    env = DummyVecEnv([lambda: StockTradingEnv(df_test)])
    model = PPO.load(path=model_path, env=env)
    obs = env.reset()
    score = 0
    for i in range(len(df_test) - 1):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        profit = env.render()
        day_profits.append(profit)
        score += rewards
        if done:
            break
    return day_profits


def modelTest2(stock_file_test, model):
    """
    模型预测
    :param stock_file_test:
    :param model:
    :return:
    """
    day_profits = []
    df_test = pd.read_csv(stock_file_test)
    env = DummyVecEnv([lambda: StockTradingEnv(df_test)])
    obs = env.reset()
    for i in range(len(df_test) - 1):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        profit = env.render()
        day_profits.append(profit)
        if done:
            break
    return day_profits


def find_file(path, name):
    """
    根据文件名name查找path下该文件的路径
    :param path:
    :param name:
    :return:
    """
    # print(path, name)
    for root, dirs, files in os.walk(path):
        for fname in files:
            if name in fname:
                return os.path.join(root, fname)


def train_and_test(stock_code, start_date, end_date, result_profit_path, reTrain=False):
    """
    股票交易，流程：训练或者获取模型->模型预测->返回结果
    :param code:  股票代码
    :param new_model: 是否重新训练模型
    :return:
    """
    try:
        # 训练模型
        model_path = "./result/model/" + stock_code
        modelTrain(stock_code=stock_code, start_date=start_date, end_date=end_date, model_path=model_path, reTrain=reTrain)
        print(f"模型准备，当前股票：{stock_code}")
        # 加载模型进行预测
        day_profits = modelTest(stock_code=stock_code, start_date=end_date, end_date='20230101', model_path=model_path)
        print(f"模型测试，当前股票：{stock_code}")
        # 追加到测试结果csv
        day_profits_df = pd.DataFrame({'stock': stock_code, 'profit': day_profits[-1]}, index=[0])
        create_or_append_to_csv(df=day_profits_df, file_path=result_profit_path, newFile=False)
        # draw_profits_for_stock(daily_profits=day_profits, stock_code=stock_code)
        print(f"完成测试，当前股票：{stock_code}")
    except Exception as e:
        traceback.print_exc()
        raise
    return day_profits


def stock_trade2(code):
    stock_file = find_file('./stockdata/train', str(code))
    day_profits = []
    df = pd.read_csv(stock_file)
    df = df.sort_values('date')

    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: StockTradingEnv(df)])

    model = PPO2(MlpPolicy, env, verbose=0, tensorboard_log='./log', gamma=0.95, n_steps=20, learning_rate=2.5e-2)
    model.learn(total_timesteps=int(1e4))

    df_test = pd.read_csv(stock_file.replace('train', 'test'))

    env = DummyVecEnv([lambda: StockTradingEnv(df_test)])
    obs = env.reset()
    for i in range(len(df_test) - 1):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        profit = env.render()
        day_profits.append(profit)
        if done:
            break
    return day_profits


def draw_profits_for_stock(daily_profits, stock_code):
    """
    单只股票收益折线图
    :param daily_profits:
    :return:
    """
    fig, ax = plt.subplots()
    ax.plot(daily_profits, '-o', label=stock_code, marker='o', ms=10, alpha=0.7, mfc='orange')
    ax.grid()
    plt.xlabel('step')
    plt.ylabel('profit')
    ax.legend(prop=font)
    plt.show()
    # plt.savefig(f'./img/{stock_code}.png')


def analysis_profits(resultsList):
    is_profit = [p[-1] for p in resultsList]
    len(is_profit)
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    font = fm.FontProperties(fname='./font/wqy-microhei.ttc')
    labels = 'Profit', 'Loss', '0'
    sizes = [0, 0, 0]
    for p in is_profit:
        if p > 0:
            sizes[0] += 1
        if p < 0:
            sizes[1] += 1
        else:
            sizes[2] += 1

    explode = (0.1, 0.05, 0.05)
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')
    plt.legend(prop=font)
    plt.show()
    plt.savefig('./img/profits.png')

    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    font = fm.FontProperties(fname='./font/wqy-microhei.ttc')
    n_bins = 150

    fig, axs = plt.subplots()
    axs.hist(is_profit, bins=n_bins, density=True)
    plt.savefig('./img/profits_hist.png')


def removeAllFiles(basePath):
    """
    删除目录下的所有文件，一层目录，不递归
    :param basePath:
    :return:
    """
    if os.path.exists(basePath):
        files = os.listdir(basePath)
        for file in files:
            if os.path.isfile(basePath + "/" + file):
                os.remove(basePath + "/" + file)


def removeFile(filePath):
    """
    删除文件
    :param basePath:
    :return:
    """
    if os.path.exists(filePath):
        os.remove(filePath)


def create_or_append_to_csv(df, file_path, newFile=False):
    """
    创建或追加，不存在则创建，存在则追加
    :param df:
    :param file_path:
    :return:
    """
    if os.path.exists(file_path):
        if newFile:
            os.remove(file_path)
        df.to_csv(file_path, mode='a', header=False, index=False, encoding='utf_8_sig')  # 判断一下file是否存在 > 存在：追加 / 不存在：保持
    else:
        df.to_csv(file_path, header=True, index=False, encoding='utf_8_sig')  # 判断一下file是否存在 > 存在：追加 / 不存在：保持


def get_stock_info_by_date(date='20221104'):
    conn = mysqlUtils.connect_database(host, port, user, password, database)
    codeSql = f"""
    select DISTINCT 
        t1.`code`,t1.`name` 
    FROM
        quant_stock_code_craw t1
    join(select DISTINCT `code` from quant_stock_data where `date`='20221104') t2 on t1.`code`=t2.`code` 
    join(select DISTINCT `code` from quant_stock_data where `date`='20100104') t3 on t2.`code`=t3.`code` 
    order by `code` 
    limit 100
    """
    stockInfo = mysqlUtils.read_data(conn, codeSql)
    conn.close()
    return stockInfo


if __name__ == '__main__':
    result_profit_path = "result_is_profit/result_profit.csv"
    stock_info_df = get_stock_info_by_date()
    pool = futures.ProcessPoolExecutor(max_workers=4)
    all_task = []
    for index, row in stock_info_df.iterrows():
        stock_code = row["code"]
        stock_name = row["name"]
        if ("ST" in stock_name) or "退" in stock_name:
            continue
        # train_and_test(stock_code=stock_code, start_date='20000101', end_date='20220101', result_profit_path=result_profit_path, reTrain=False)
        task = pool.submit(train_and_test, stock_code, '20000101', '20220101', result_profit_path, False)
        all_task.append(task)
    futures.wait(fs=all_task, return_when=futures.ALL_COMPLETED)
    sys.exit()