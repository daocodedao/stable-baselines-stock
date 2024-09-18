import warnings

import pymysql

import pandas as pd
from dbutils.pooled_db import PooledDB, PooledSharedDBConnection

warnings.filterwarnings('ignore')

mysql_conf = {
    'host': '127.0.0.1',
    'user': 'root',
    'passwd': '123456',
    'db': 'zeus',
    'port': 3306
}
# pool = PooledDB(pymysql, 20, **mysql_conf)

def connect_database(host, port, user, password, database):
    dbInstance = pymysql.connect(host=host, port=port, user=user, password=password, database=database)
    return dbInstance

#
# def connection_borrow():
#     return pool.connection()
#
#
# def connection_back(conn: PooledSharedDBConnection):
#     conn.close()


def read_data(dbInstance, sql):
    cursor = dbInstance.cursor()  # 使用cursor()方法获取用于执行SQL语句的游标
    cursor.execute(sql)  # 执行SQL语句
    data = cursor.fetchall()
    # 下面为将获取的数据转化为dataframe格式
    columnDes = cursor.description  # 获取连接对象的描述信息
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]  # 获取列名
    df = pd.DataFrame([list(i) for i in data], columns=columnNames)  # 得到的data为二维元组，逐行取出，转化为列表，再转化为df
    """
    使用完成之后需关闭游标和数据库连接，减少资源占用,cursor.close(),db.close()
    db.commit()若对数据库进行了修改，需进行提交之后再关闭
    """
    cursor.close()
    return df


if __name__ == '__main__':
    host = "127.0.0.1"
    port = 3306
    user = "root"
    password = '123456'
    database = 'zeus'

    dbInstance = connect_database(host, port, user, password, database)
    sqlStr = "select * from quant_stock_data where `code`='000001' order by `date`"
    df = read_data(dbInstance, sqlStr)
    print(df.tail())
    df = read_data(dbInstance, sqlStr)
    dbInstance.close()