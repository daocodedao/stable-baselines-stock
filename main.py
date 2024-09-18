# http://baostock.com/baostock/index.php/%E9%A6%96%E9%A1%B5
import os
import pickle
import pandas as pd
# from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from rlenv.StockTradingEnv0 import StockTradingEnv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font = fm.FontProperties(fname='font/wqy-microhei.ttc')
# plt.rc('font', family='Source Han Sans CN')
plt.rcParams['axes.unicode_minus'] = False


def modelTrain(stock_file, model_path, reTrain=False):
    if os.path.exists(model_path) and (not reTrain):
        return

    print(f"训练模型，当前股票：{stock_file}")

    df = pd.read_csv(stock_file)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df.sort_values('date')

    # The algorithms require a vectorized environment to run
    env = StockTradingEnv(df)

    # model = PPO("MlpPolicy", env, verbose=0, tensorboard_log='./log')
    model = PPO("MlpPolicy", env=env, verbose=1)
    model.learn(total_timesteps=10_000)

    model.save(model_path)



def modelTest(stock_file,  model_path):

    df_test = pd.read_csv(stock_file.replace('train', 'test'))
    df_test.dropna(inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    df_test = df_test.sort_values('date')
    
    env = StockTradingEnv(df_test)
    model = PPO.load(path=model_path, env=env)
    obs = env.reset()
    day_profits = []
    score = 0
    for i in range(len(df_test) - 1):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        profit = env.render(mode="human")
        day_profits.append(profit)
        score += rewards
        if done:
            break
    return day_profits

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


def train_and_test(stock_code):
    model_path = "./result/model/" + stock_code
    train_stock_file = find_file('./stockdata/train', str(stock_code))
    print(f"模型准备，当前股票：{stock_code}")
    modelTrain(stock_file=train_stock_file, model_path=model_path)
    
    test_stock_file = find_file('./stockdata/test', str(stock_code))
    print(f"模型测试，当前股票：{stock_code}")
    day_profits = modelTest(stock_file=test_stock_file, model_path=model_path)

    day_profits_df = pd.DataFrame({'stock': stock_code, 'profit': day_profits[-1]}, index=[0])
    result_profit_path = "./result/result_profit.csv"
    create_or_append_to_csv(df=day_profits_df, file_path=result_profit_path, newFile=False)
        # draw_profits_for_stock(daily_profits=day_profits, stock_code=stock_code)
    print(f"完成测试，当前股票：{stock_code}")

def find_file(path, name):
    # print(path, name)
    for root, dirs, files in os.walk(path):
        for fname in files:
            if name in fname:
                return os.path.join(root, fname)


def test_a_stock_trade(stock_code):
    stock_file = find_file('./stockdata/train', str(stock_code))

    daily_profits = stock_trade(stock_file)
    fig, ax = plt.subplots()
    ax.plot(daily_profits, '-o', label=stock_code, marker='o', ms=10, alpha=0.7, mfc='orange')
    ax.grid()
    plt.xlabel('step')
    plt.ylabel('profit')
    ax.legend(prop=font)
    # plt.show()
    plt.savefig(f'./img/{stock_code}.png')


def multi_stock_trade():
    start_code = 600000
    max_num = 3000

    group_result = []

    for code in range(start_code, start_code + max_num):
        stock_file = find_file('./stockdata/train', str(code))
        if stock_file:
            try:
                profits = stock_trade(stock_file)
                group_result.append(profits)
            except Exception as err:
                print(err)

    with open(f'code-{start_code}-{start_code + max_num}.pkl', 'wb') as f:
        pickle.dump(group_result, f)


if __name__ == '__main__':
    # multi_stock_trade()
    train_and_test('sh.600755')
    # ret = find_file('./stockdata/train', '600036')
    # print(ret)

