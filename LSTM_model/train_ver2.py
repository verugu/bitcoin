#ファイナンス機械学習で学んだことを使って再構築する。

#使用データ...2017-21の5年間の5分足データとする。CUSUMフィルタによってダウンサンプリングを行う。
#ラベル...トリプルバリア法を適用し、リターンの符号をラベルとする。標本はリターンの絶対値で重み付けを行う。
#特徴量...分数差分次を使ってみる。他は引き続きlogや様々な期間での指数加重平均や指数加重標準偏差なども用いてみる。
#モデル...ラベルを予測する1次モデルとしてLSTMを使用。メタラベリングを施しベットサイズを学習する２次モデルにはQ-learnの強化学習を用いる予定で、こちらもLSTMを用いる。


import torch
import torch.nn as nn
from torch.nn.modules import dropout
#import torch.nn.functional as F
#import torch.optim as optim
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd
import copy



#特徴量を作成する。必要になると思われるパラメータ群は
#csv_path...生データがあるパス
#batch_size...バッチサイズ

#サンプリング
#ret_delta...ボラリティを何次リターンをもとに計算するか
#vol_term...どれくらいの期間を用いてその時点のボラリティを計算するか
#train_prop...訓練データの割合

#特徴量作成
#past_num...過去どれくらいの期間を予測に使うか
#transform...特徴量の作成方法。リストとして渡す。各々の特徴量の作成に関するパラメータはmain内で与える。ここで与えるのはそれらの「リスト」。

#ラベル作成
#vertical_delta...垂直バリアの期間
#ptSl...水平バリアにおいて窓幅にかけられる利食いと損切の割合。

class RateDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, batch_size, ret_delta, vol_term, train_prop, past_num, vertical_delta, ptSl=[1,1], transform=None):
        self.raw_data = pd.read_csv(csv_path, index_col=0)
        self.raw_data.index = pd.to_datetime(self.raw_data.index)
        self.batch_size = batch_size
        self.ret_delta = ret_delta
        self.vol_term = vol_term
        self.train_prop = train_prop
        self.past_num = past_num
        self.vertical_delta = vertical_delta
        self.ptSl = ptSl
        self.transform = transform
        
        #今回使うデータをCUSUMフィルタによってサンプリング
        vol = getVol(self.raw_data, self.ret_delta, self.vol_term) #ボラリティ計算。
        t_events = getTEvents(self.raw_data, vol) #各時点のボラリティに基づいてイベント抽出
        print('サンプル数:', t_events.size)
        
        #トリプルバリアによってラベル作成
        vertical_barrier = t_events + self.vertical_delta #垂直バリア
        events = pd.DataFrame(vertical_barrier)
        events.index = t_events
        events.columns = ['t1']
        events['trgt'] = vol
        events = events.dropna()



#ローリング指数加重標準偏差。pandasのewm.std()を期間が指定できるように改良。
def ewm_std(close, term, span):

    df = copy.deepcopy(close)
    N = span
    a = 2./(1+N)
    time_delta = term
    term_num = df[:df.index[0] + time_delta].shape[0] #time_deltaの分だけの期間のデータ個数

    df_raw = np.pad(df.values[:,0], [term_num-1, 0])
    stdcalc = []

    # Get weights: w
    w = (1-a)**np.arange(term_num) # This is reverse order to match Series order ←np.convolveがクソみたいな仕様なので逆向きやめた。
    w_inv = w[::-1]
    sum_w = np.sum(w)

    # Calculate exponential moving average
    ewma = np.convolve(df_raw, w, mode='valid') / sum_w


    # Calculate bias
    bias = np.sum(w)**2 / (np.sum(w)**2 - np.sum(w**2))

    # Calculate exponential moving variance with bias
    stdcalc = np.array([bias * np.sum(w_inv * (df_raw[i : i+term_num] - ewma[i])**2) / sum_w for i in range(df.shape[0])])
    
    #ewmvar = bias * np.sum(w * (z - ewma)**2) / np.sum(w)

    
    stdcalc = pd.DataFrame(np.sqrt(stdcalc), index=df.index, columns=['vol'])

    return stdcalc


#平均バージョン
def ewm_mean(close, term, span=100):

    df = copy.deepcopy(close)
    N = span
    a = 2./(1+N)
    time_delta = term
    term_num = df[:df.index[0] + time_delta].shape[0] #time_deltaの分だけの期間のデータ個数

    df_raw = np.pad(df.values[:,0], [term_num-1, 0])

    # Get weights: w
    w = (1-a)**np.arange(term_num) # This is reverse order to match Series order ←np.convolveがクソみたいな仕様なので逆向きやめた。
    sum_w = np.sum(w)

    # Calculate exponential moving average
    ewma = np.convolve(df_raw, w, mode='valid') / sum_w

    ewma_df = pd.DataFrame(ewma, index=df.index, columns=['mean'])

    return ewma_df



def getVol(close, ret_delta, term, span0=100):
    df0 = close.index.searchsorted(close.index-ret_delta, side='right')
    df0 = df0[df0>0]
    df0 = pd.Series(close.index[df0-1], index=close.index[close.shape[0] - df0.shape[0]:])

    df0 = close.loc[df0.index]/close.loc[df0.values].values - 1
    df0 = ewm_std(df0, term=term, span=span0)
    return df0.dropna()




#CUSUMフィルタ。その時点のボラリティから動的閾値を計算する方式に変更。トリプルバリアも動的閾値だし、こっちもそうするのが普通だという考え。
def getTEvents(gRaw, vol):

    tEvents, sPos, sNeg = [], 0, 0
    diff = gRaw.diff().dropna()/gRaw.values[:-1] #そもそも絶対差(diff())のみじゃスケール依存なのでリターンみたいに割るべき
    np_vol = np.array(vol)[:, 0]

    for p, i in enumerate(vol.index):
        sPos, sNeg = max(0, sPos + diff.loc[i].values), min(0, sNeg + diff.loc[i].values)
        h = np_vol[p]
        
        #hが低すぎると過剰イベントに、高すぎると過疎になるので防止
        if h < 0.005:
            h = 0.005
        
        elif h > 0.008:
            h = 0.008
        
        if sNeg < -h:
            sNeg = 0
            tEvents.append(i)
        elif sPos > h:
            sPos = 0
            tEvents.append(i)
    
    return pd.DatetimeIndex(tEvents)