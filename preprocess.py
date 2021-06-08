# 取早上8點到下午5點
# 把 .txt資料轉成 .csv
# 把都是0的data抓出來

import numpy as np
import pandas as pd
import os
import glob

# column_names_1=np.arange(13) #(1到13) column的label
column_names_1 = np.arange(14)  # (1到14) column的label (QCNN用)

# column_names_2=['date','time','tep','kw','li'] #column的label
column_names_2 = ['date', 'time', 'tep', 'li', 'kw']  # column的label (QCNN用)

# path1=r'C:\\Users\\asus1\\master1\\sun_data_input' #讀檔的路徑
path1 = r'./txt/'  # 讀檔的路徑

file1 = glob.glob(os.path.join(path1, "1Solar_INV*.txt"))  # 1Solar201712* 讀檔的檔名  *=>代表(01,02....) 會讀取資料夾裡有命名到的數字

loss_data = pd.DataFrame([], dtype='int32')  # 空的loss_data 之後如果偵測到符合條件的日期數據會 concat進來
temp = pd.DataFrame(np.ones((1, 3)) * 0, dtype='int32',
                    columns=['zero', 'loss', 'loss_count'])  # 初始值維[1,1,1] 用來儲存'zero','loss','loss_count'的資料

for f in file1:
    # path2='C:\\Users\\asus1\\master1\\sun_data_ok\\' #輸出的資料路徑
    path2 = './data/'  # 輸出的資料路徑
    if not os.path.isdir(path2):  # 檢查資料夾，若無則新增
        os.mkdir(path2)
    out = pd.read_csv(f, sep=',')  # 讀檔 以逗號 為分隔
    out.columns = column_names_1  # 將讀入的檔案 column的label改為column_names_1
    # out=out.drop([0,3,4,6,7,8,9,10,],axis=1) #把第[0,3,4,6,7,8,9,10,]行去掉
    # out=out.ix[:,[1,2,11,12,5]] #以[1(date),2(time),11(tep),5(kw),12(li)] 順續去排列

    out = out.drop([0, 4, 5, 7, 8, 9, 10, 11, ], axis=1)  # 把第[0,3,4,6,7,8,9,10,]行去掉 (QCNN用)
    out = out.loc[:, [2, 3, 12, 13, 6]]  # 以[2(date),3(time),12(tep),13(li),6(kw)] 順續去排列 (QCNN用)

    out.iloc[:, 2:4] = out.iloc[:, 2:4] + 0.000001  # 把tep kw 兩行都加上0.000001 確保沒有零列
    out.columns = column_names_2  # column的label改為column_names_2
    index_names_1 = out.loc[:, 'time']  # 把 'time'那行的數據 讀出 => index_names_1
    out.index = index_names_1  # 把index的label 以'time'這行的時間命名
    out = out.loc['08:01:00':'17:00:00', :]  # 用index的label 去保留 要的時間
    count_zero = out.loc[:, 'li'].value_counts(0)  # value_counts(0) 去計算 'li'那行每個數據的次數(次數多的會在最上面) 會以DataFrame的形式當回傳值
    if count_zero.index[0] != 0:  # 如果0不是最多次的
        path2 = path2 + str(out.iloc[0, 0]) + '.csv'  # 路徑加上 out文件的(0,0)位置的值 =>就是日期 當輸出路徑
        out.to_csv(path2, index=False)  # 輸出資料  index=False=>index的label去掉
    else:  # 如果0是最多次的
        path2 = path2 + str(out.iloc[0, 0]) + '.csv'
        out.to_csv(path2, index=False)
        temp.iloc[0, 0] = out.iloc[0, 0]  # temp(0,0)位置等於 日期
        loss_data = pd.concat([loss_data, temp], axis=0,
                              ignore_index=True)  # concat 結合[loss_data,temp] 以axis=0上下的方向合併 ,ignore_index=True 忽略index 合併
    print(str(out.iloc[0, 0]), out.shape[0])  # print out.iloc[0,0]日期  和out.shape[0]列數 (檢查資料有沒有少)
    if out.shape[0] != 540:  # (正常661筆) 如果不等於661
        temp.iloc[0, 1] = out.iloc[0, 0]  # temp.iloc[0,1]=>日期
        temp.iloc[0, 2] = (540 - out.shape[0])  # 661-out的列數
        loss_data = pd.concat([loss_data, temp], axis=0, ignore_index=True)  # 把temp concat到loss_data
    temp = pd.DataFrame(np.ones((1, 3)) * 0, dtype='int32', columns=['zero', 'loss', 'loss_count'])  # 每一次都初始化temp
# path3='C:\\Users\\asus1\\master1\\sun_data_ok\\lose.csv' #loss_data 的路徑檔名
path3 = './lose.csv'  # loss_data 的路徑檔名
loss_data.to_csv(path3, index=False)  # 輸出loss_data