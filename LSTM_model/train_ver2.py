#ファイナンス機械学習で学んだことを使って再構築する。

#使用データ...2017-21の5年間の5分足データとする。CUSUMフィルタによってダウンサンプリングを行う。
#ラベル...トリプルバリア法を適用し、リターンの符号をラベルとする。標本はリターンの絶対値で重み付けを行う。
#特徴量...分数差分次を使ってみる。他は引き続きlogや様々な期間での指数加重平均や指数加重標準偏差なども用いてみる。
#モデル...ラベルを予測する1次モデルとしてLSTMを使用。メタラベリングを施しベットサイズを学習する２次モデルにはQ-learnの強化学習を用いる予定で、こちらもLSTMを用いる。



from operator import imod
from unicodedata import bidirectional
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.modules import dropout
#import torch.nn.functional as F
#import torch.optim as optim
from torch.optim import SGD
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd
from pandas import Timedelta
import copy

import multiprocessing

from tcn import TemporalConvNet



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


#交差検証について
#組み合わせパージング交差検証(cvcs)を用いてみる。10個にデータを分割し、2個をテストに使う。この場合経路は10C2=45で、45/9=5通りの経路。

class RateDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, batch_size, ret_delta, vol_term, past_term, vertical_delta, max_term, feature_path, label_path, feature_idx,
                 feature_load=True, label_load=True, ptSl=[1,1], transform=None):
        
        self.raw_data = pd.read_csv(csv_path, index_col=0)
        self.raw_data.index = pd.to_datetime(self.raw_data.index)
        self.rate_data = self.raw_data.values[:, 0] #価格列
        self.batch_size = batch_size
        self.past_term = past_term #過去どれくらいのデータをlstmに突っ込むかもtimedelta型にする
        self.past_num = np.where(self.raw_data.index==self.raw_data.index[0]+self.past_term)[0][0] #past_termを過去何点のデータを使うかというデータに変換したもの
        self.max_term = max_term
        self.transform = transform
        self.feature_idx = feature_idx

        self.train_size = 0
        self.test_size = 0
        

        
        if label_load:
            self.weighted_labels = np.load(label_path, allow_pickle=True)
            self.len = self.weighted_labels.shape[0]
            print('使うデータ数:', self.len)
        
        else:
            #今回使うデータをCUSUMフィルタによってサンプリング
            vol = getVol(self.raw_data, ret_delta, vol_term) #ボラリティ計算。
            t_events = getTEvents(self.raw_data, vol) #各時点のボラリティに基づいてイベント抽出
            print('サンプル数:', t_events.size)
            
                
            
            #トリプルバリアによってラベル作成。ラベル作成を先にやってるのは、後の特徴量とtimeindexを楽に合わせたいから。
            vertical_barrier = t_events + vertical_delta #垂直バリア
            events = pd.DataFrame(vertical_barrier)
            events.index = t_events
            events.columns = ['t1']
            events['trgt'] = vol
            events = events.dropna()

            tp_events = applyPtSlOnT1(self.raw_data, events, ptSl)
            t1_events = pd.DataFrame(tp_events.min(axis=1), columns=['t1']) #最初にトリプルバリアに触れた時の時刻群
            t1_events = t1_events[t1_events.index >= self.raw_data.index[0]+max_term] #過去past_term分+リターンボラリティ分の計算で使用する分は消しておく。
            t1_events = del_outliers(self.raw_data, t1_events, vertical_delta, past_term) #coincheckが止まって価格変動がない時を除外

            labels = getBins(t1_events, self.raw_data).dropna()

            #標本をリターンで重み付け（リターンが小さいところは低いウェイト）
            labels['weight'] = 1.0
            labels['weight'][abs(labels['ret'])<0.01] = 0.8
            labels['weight'][abs(labels['ret'])<0.001] = 0.4
            #labels['weight'][labels['weight']<=0.] = 1.0
            #1%以下で0.8, 0.1%以下で0.4だったっけ
            
            #bin, ret, weightから成るDataFrame
            self.labels = labels

            bin = abs(labels.bin.values-1)
            weig = labels['weight'].values
            weighted_labels = np.identity(2)[-bin.astype('int64')]*weig[:, None]
            
            #timeインデックスのあるdfではなくただのndarray。[0.8, 0]なら買いラベルで0.8のウェイトということ
            self.weighted_labels = weighted_labels

            np.save(label_path, self.weighted_labels)


            
            t_events = self.labels.index #最終的に訓練/テストに使うデータの時刻群
            print('最終的な使用データ数:', len(t_events))

            self.len = len(t_events)

        #特徴量作成。最後に書きます。
        #時系列データ...rawデータ（正規化）、対数データ、分数差分時データ、ボラリティ（期間など変更)、加重平均線
        #LSTMを用いるわけだし、全て時系列データにする。
        
 
        #特徴量作成時間かかるのでnpy形式で保存してロード。
        if feature_load:
            self.features = np.load(feature_path, allow_pickle=True).tolist()
        
        else:

            #forで回すしマルチプロセッシングでやった方が良さげか(forで回さない方法が思いつかない)。特徴量には対象時刻をラベルとして与える
            with multiprocessing.Pool(processes=6) as pool:
                features_value = pool.map(self.make_features, t_events)
            
            #時刻をkey、特徴量をvalueとして作成(分かりやすい)
            self.features = dict(zip(t_events, features_value))

            np.save(feature_path, self.features)


        
    
    #マルチプロセッシングで使う用
    def make_features(self, time):
        features = np.zeros([self.past_num, len(self.transform)])
        
        #時刻からその位置インデックスを取得
        t_idx = np.where(self.raw_data.index == time)[0][0]

        for i, transform in enumerate(self.transform):
            #時刻の位置インデックスとpast_numから、ndarrayだけで計算できるようにする
            feature = transform(self.raw_data, t_idx, self.past_num)
            features[:, i] = feature
        
        return features
    
    
    def makeBatch(self, test_group):
        
        keys = self.features.keys()
        keys = np.array(list(keys)).astype('datetime64[s]')


        train_indices, test_indices = self.split_cvcs(keys, test_group, self.max_term, div=10)
        self.train_size = len(train_indices)
        self.test_size = len(test_indices)

        #ある地点より過去のデータのみで学習し、testはその後のデータを用いてやるために、ランダムにはしない
        train_set = torch.utils.data.Subset(self, train_indices)
        test_set = torch.utils.data.Subset(self, test_indices)


        train_dataloader = torch.utils.data.DataLoader(train_set, self.batch_size, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_set, self.batch_size, shuffle=False)

        return train_dataloader, test_dataloader
    
    def split_cvcs(self, times, test_group, term, div=10):
        length = len(times)
        unit = length//div #1グループあたりのデータ数
        indices = np.arange(length)
        test1 = indices[unit*test_group[0] : unit*(test_group[0]+1)]
        test2 = indices[unit*test_group[1] : unit*(test_group[1]+1)]
        test_indices = np.concatenate([test1, test2])
        
        #term分重ならないように訓練データを消す
        t1_start = times[test1[0]]-term; t1_end = times[test1[-1]]+term
        t2_start = times[test2[0]]-term; t2_end = times[test2[-1]]+term

        train_indices = np.where((times<=t1_start) | 
                                ((times>=t1_end) & (times<=t2_start)) |
                                (times>=t2_end))[0]
        return train_indices, test_indices
    
    def get_datasize(self):
        return self.train_size, self.test_size


    def __len__(self):
        return self.len

    #ここで特徴量はそのまま返して、ラベルは少し工夫をする。1,0の分類として扱いつつ、ラベルはリターンによって重み付けをしたい。
    #重み付けというのはすなわち損失をその分増やすことなので、loss関数を自作しなければならないか？
    def __getitem__(self, idx):
        return list(self.features.values())[idx][:, self.feature_idx], self.weighted_labels[idx]


#ローリング指数加重標準偏差。pandasのewm.std()を期間が指定できるように改良。
def ewm_std(close, term, span):

    df = copy.deepcopy(close)
    N = span
    a = 2./(1+N)
    time_delta = term
    term_num = df[:df.index[0] + time_delta].shape[0] #time_deltaの分だけの期間のデータ個数
    
    #足りない分は0埋めすることで回避
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
    '''
    本書のサンプルコード
    '''
    tEvents, sPos, sNeg = [], 0, 0
    diff = gRaw.diff().dropna()/gRaw.values[:-1] #そもそも絶対差(diff())のみじゃスケール依存なのでリターンみたいに割るべき
    offset = np.where(diff.index==vol.index[0])[0][0]
    diff = diff.values[:, 0]
    np_vol = np.array(vol)[:, 0]
    
    for p, i in enumerate(vol.index):
        sPos, sNeg = max(0, sPos + diff[offset+p]), min(0, sNeg + diff[offset+p])
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



#トリプルバリア
def applyPtSlOnT1(close, events, ptSl, molecule=None):
    # t1(イベント終了)前に行われた場合は、ストップロス/利食いを実施
    events_ = copy.deepcopy(events) #.loc[molecule]
    out = events_[['t1']].copy(deep=True)
    
    if ptSl[0] > 0:
        pt = ptSl[0] * events_['trgt']
    else:
        pd.Series(index=events.index)
    
    if ptSl[1] > 0:
        sl = -ptSl[1] * events_['trgt']
    else:
        pd.Series(index=events.index)
    

    sl_values = np.zeros(events.shape[0], dtype='datetime64[s]')
    pt_values = np.zeros(events.shape[0], dtype='datetime64[s]')
    idx = 0
    for loc, t1 in events_['t1'].iteritems(): #(index, value)でiter
        df0 = close[loc:t1] # 価格経路
        df0 = (df0 / close.loc[loc] - 1) #* events_.at # リターン
        
        loss = sl[loc]
        profit = pt[loc]
        
        
        #損益幅が大きすぎると機能しないので、最低でも1.0%下がるのはやめたい。つまり最大幅を0.01にする
        if loss < -0.02:
            loss = -0.02

        #利益は最低でも0.5%は欲しいし、1.0%で十分
        if profit > 0.01:
            profit = 0.01
        
        elif profit < 0.005:
            profit = 0.005

        # ストップロスの最短タイミング
        sl_value = df0[df0.values <= loss].index.min()
        if sl_value is pd.NaT:
            sl_values[idx] = np.datetime64('NaT')
        else:
            sl_values[idx] = sl_value
        
        pt_value = df0[df0.values >= profit].index.min()
        # 利食いの最短タイミング
        if pt_value is pd.NaT:
            pt_values[idx] = np.datetime64('NaT')
        else:
            pt_values[idx] = pt_value
        
        idx += 1
    
    out['sl'] = sl_values
    out['pt'] = pt_values
    
    return out

def del_outliers(data, events, vertical, past):
    t_events = events
    rate = data.values[:,0]
    t = data.index
    t_l = []

    count = 0
    for i in range(data.shape[0]-1):
        if rate[i] == rate[i+1]:
            if count == 0:
                start = t[i]
            
            count += 1
        
        #1時間以上ストップしてるなら除外したい
        elif count > 12:
            count = 0
            end = t[i]
            t_l.append([start-vertical,end+past]) #特徴量やラベリングに影響する部分を除外
        
        else:
            count = 0
    
    for times in t_l:
        start = times[0]
        end = times[1]
        t_events = t_events[(t_events.index<start) | (t_events.index>end)]

    
    return t_events

#ラベリング
def getBins(events, close):
    #1) イベント発生時の価格
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    
    #2) outオブジェクトの生成
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1
    out['bin'] = np.sign(out['ret'])
    out.loc[out['bin']<=0, 'bin'] = 0

    return out



#生データを最大最小で正規化
class Standardize:
    def __call__(self, raw_data, t_idx, past_num):
        rate_data = copy.deepcopy(raw_data.values[:, 0])
        rate_max_min = std_max_min(rate_data[t_idx-past_num+1 : t_idx+1])
        return rate_max_min

#log変換
class Log_scale:
    def __call__(self, raw_data, t_idx, past_num):
        rate_data = copy.deepcopy(raw_data.values[:, 0])
        log_data = np.log(rate_data[t_idx-past_num+1 : t_idx+1])
        log_data = std_max_min(log_data)
        return log_data

class Volatility:
    def __init__(self, term, span=100):
        self.term = term
        self.term_num = int(term/Timedelta(minutes=5))
        self.span = span
    
    def __call__(self, raw_data, t_idx, past_num):
        vol = ewm_std(raw_data.iloc[t_idx-past_num-self.term_num+1:t_idx+1], self.term, self.span).values[:,0]
        vol = std_max_min(vol[-past_num:])
        return vol


class RetVolatility:
    def __init__(self, ret_delota, term, span=100):
        self.ret_delta = ret_delota
        self.term = term
        self.span = span
    
    def __call__(self, raw_data, t_idx, past_num):
        vol = getVol(raw_data.iloc[:t_idx+1, :], self.ret_delta, self.term, self.span)
        
        #例えば日次リターンボラリティなら2017/1/2から始まるのでt_idxは使えない(t_idxはraw_dataでの位置インデックス)
        #ゆえに、volの中でのインデックスを再取得する必要がある（これ以外の上のtransform群はdataと同じtimeIndexになるように設計してある)
        #と思ったけど、これもt_idxまでのデータのみ渡せば-past_num:で取り出せるわ
        vol = std_max_min(vol[-past_num:])
        return vol


class Moving_average:
    def __init__(self, term, span=100):
        self.term = term
        self.term_num = int(term/Timedelta(minutes=5))
        self.span = span
    
    def __call__(self, raw_data, t_idx, past_num):
        w_average = ewm_mean(raw_data.iloc[t_idx-past_num-self.term_num+1:t_idx+1], self.term, self.span).values[:,0]
        w_average = std_max_min(w_average[-past_num:])
        return w_average

#分数差分次
class FFD:
    def __init__(self, d, thres):
        self.d = d
        self.thres = thres
        self.weights = get_weights(d, thres)
        self.len = len(self.weights)
    
    def __call__(self, raw_data, t_idx, past_num):
        rate_data = raw_data.values[:,0][t_idx-past_num-self.len+2:t_idx+1]
        ffd = np.convolve(rate_data, self.weights, mode='valid')
        ffd = std_max_min(ffd)

        return ffd



def std_max_min(array):
    max = np.nanmax(array)
    min = np.nanmin(array)
    tensor = (array - min) / (max - min)
    #tensor = digitizing(tensor, 20)
    return tensor

#データを、例えば100個の区切りで離散化しなおす。0-1のデータ前提（だるいので）
def digitizing(data, split):
    unit = 1/split
    dig_data = np.ones(data.shape[0])
    for i in range(split):
        dig_data[(data>=unit*i) & (data<unit*(i+1))] = unit*i
    
    return dig_data

def get_weights(d, thres):
    w, k = [1.], 1
    while True:
        w_ = -w[-1]/k*(d-k+1)
        if abs(w_)<thres:
            break
        
        w.append(w_)
        k += 1
    
    return np.array(w[::-1])


#CNN合った方が良さげなのでcnnベース。さらにattentionを組み合わせてみる。→時系列データではノイズにしかならないっぽい（precisionがカスだった）
class CNN_LSTM_ATTNNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout, kernel_size, batch_first=True):
        super(CNN_LSTM_ATTNNet, self).__init__()
        self.hidden_size = hidden_size

        self.conv1 = nn.Conv1d(input_size, input_size, kernel_size, padding=kernel_size//2, padding_mode='reflect') #[inputチャンネル数, outputチャンネル数, kernel_size]が関数入力
        self.pool = nn.AvgPool1d(2, stride=2) #半分にする
        self.tanh = nn.Tanh()
        #self.conv2 = nn.Conv1d(1, 1, kernel_size, padding=kernel_size//2, padding_mode='reflect')
        #self.relu = nn.ReLU()
        
        #畳み込み層
        self.cnn = nn.Sequential(self.conv1, self.pool, self.tanh)

        #リカレントネットワーク層
        self.rnn = nn.LSTM(input_size = input_size,
                           hidden_size = hidden_size,
                           batch_first = batch_first,
                           bidirectional = True)
        
        self.norm = nn.BatchNorm1d(num_features=hidden_size)
        self.drop = nn.Dropout(dropout)
        
        self.attn_weight = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.ReLU(True), nn.Linear(hidden_size//2, 1)) #隠れ状態をそのsequenceのattentionウェイトにする。
        
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.BatchNorm1d(num_features=hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        #self.norm2 = nn.BatchNorm1d(num_features=output_size)
        #self.linear = nn.Linear(hidden_size, output_size)

        

        #全結合層。norm2のせいで[正、負]みたいな感じになってたので消す。代わりにsoftmaxということで。
        self.linear = nn.Sequential(self.linear1, self.norm1, self.linear2) #self.norm2)


        #hidden_state = torch.zeros(1, batch_size, hidden_size).to(device)
        #cell_state = torch.zeros(1, batch_size, hidden_size).to(device)
        #self.hidden = (hidden_state, cell_state)
        nn.init.xavier_normal_(self.rnn.weight_ih_l0)
        nn.init.orthogonal_(self.rnn.weight_hh_l0)
        
       

    def forward(self, inputs):

        #cnn→lstm層
        h = self.cnn(inputs.transpose(1,2))
        h, _ = self.rnn(h.transpose(1,2))
        h = h[:, :, :self.hidden_size] + h[:, :, self.hidden_size:] #全てのsequenceでの隠れ状態(双方向なので２つ)を足し合わせる(tanh+tanh)。[batch, sequence, hidden]
        #h = self.norm(h) #重み計算のためにdropは後で
        
        #[batch*sequence, hidden]をattn_weightにかける→[batch*seq, 1]となるので[batch, seq]にreshapeしてsequenceごとの重み獲得。
        batch_size = h.size(0)
        h = self.drop(h)
        attn = self.attn_weight(h.reshape(-1, self.hidden_size))
        attn = torch.softmax(attn.reshape(batch_size, -1), dim=1).unsqueeze(2)

        #h(各sequenceでの隠れ状態を持つ)にattnという重みをかけて足すことでattention。[batch, seq, hid] * [batch, seq]→[batch, hidden]
        output = (h * attn).sum(dim=1)
        output = self.norm(output) #batch正規化
    
        output = self.linear(output) #[batch, 2]。[0.24, 0.76]みたいなのがbatch行並んでいる
        
        #pytorchのクロスエントロピーはsoftmaxとるのでここで処理する必要なし。
        if self.training == False:
            output = torch.softmax(output, dim=1)
        #output = self.linear2(output)
        #output = self.linear3(output)

        return output

#モデル定義
class CNN_BiLSTMNet1(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout, kernel_size, batch_first=True):
        super(CNN_BiLSTMNet1, self).__init__()
        self.hidden_size = hidden_size

        self.conv1 = nn.Conv1d(input_size, input_size, kernel_size, padding=kernel_size//2, padding_mode='reflect') #[inputチャンネル数, outputチャンネル数, kernel_size]が関数入力
        self.pool = nn.MaxPool1d(2, stride=2) #半分にする
        self.tanh = nn.Tanh()
        #self.conv2 = nn.Conv1d(1, 1, kernel_size, padding=kernel_size//2, padding_mode='reflect')
        #self.relu = nn.ReLU()
        
        #畳み込み層
        self.cnn = nn.Sequential(self.conv1, self.pool, self.tanh)

        #リカレントネットワーク層
        self.rnn = nn.LSTM(input_size = input_size,
                           hidden_size = hidden_size,
                           batch_first = batch_first,
                           bidirectional = True)
        
        self.norm = nn.BatchNorm1d(num_features=2*hidden_size)
        
        self.drop = nn.Dropout(dropout)
        
        
        self.linear1 = nn.Linear(hidden_size*2, hidden_size) #双方向lstmは[forward層の最終隠れ状態, back層の最終隠れ状態]が出力なので２倍の長さ
        self.norm1 = nn.BatchNorm1d(num_features=hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        #self.norm2 = nn.BatchNorm1d(num_features=output_size)
        #self.linear = nn.Linear(hidden_size, output_size)

        

        #全結合層。norm2のせいで[正、負]みたいな感じになってたので消す。代わりにsoftmaxということで。
        self.linear = nn.Sequential(self.linear1, self.norm1, self.linear2) #self.norm2)


        #hidden_state = torch.zeros(1, batch_size, hidden_size).to(device)
        #cell_state = torch.zeros(1, batch_size, hidden_size).to(device)
        #self.hidden = (hidden_state, cell_state)
        nn.init.xavier_normal_(self.rnn.weight_ih_l0)
        nn.init.orthogonal_(self.rnn.weight_hh_l0)
        
       

    def forward(self, inputs):
        h = self.cnn(inputs.transpose(1,2))
        
        batch = inputs.size(0)
        h_0 = Variable(torch.zeros(2, batch, self.hidden_size)).to('cuda')
        c_0 = Variable(torch.zeros(2, batch, self.hidden_size)).to('cuda')
        h, _ = self.rnn(h.transpose(1,2), (h_0, c_0))

        h_concat = torch.concat([h[:, -1, :self.hidden_size], h[:, 0, self.hidden_size:]], dim=1)
        h_concat = self.norm(h_concat)
        h_concat = self.drop(h_concat)
        #h    = self.drop1(h)
        #h, _ = self.rnn2(h)
        #h    = self.drop2(h)
        #output = self.cnn(output[:, None, :])
    
        output = self.linear(h_concat) #[batch, 2]。[0.24, 0.76]みたいなのがbatch行並んでいる
        
        #pytorchのクロスエントロピーはsoftmaxとるのでここで処理する必要なし。
        if self.training == False:
            output = torch.softmax(output, dim=1,)
        #output = self.linear2(output)
        #output = self.linear3(output)

        return output

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, seq_len):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(seq_len, output_size)
    
    def forward(self, x):
        output = self.tcn(x.transpose(1,2))
        output = self.linear(output).squeeze(1)

        if self.training == False:  
            output = torch.softmax(output, dim=1,)

        return output

class TCN_LSTM(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, seq_len):
        super(TCN_LSTM, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.rnn = nn.LSTM(input_size = 1,
                           hidden_size = 2,
                           batch_first = True,
                           bidirectional = False)
    
    def forward(self, x):
        output = self.tcn(x.transpose(1,2))
        output, _ = self.rnn(output.transpose(1,2))
        output = output[:,-1,:]

        if self.training == False:  
            output = torch.softmax(output, dim=1,)

        return output

def calc_accuracy(predict, label):
    act_label = torch.min(label.data, 1)[1]
    pred_label = torch.min(predict.data, 1)[1]
    acc_tensor = act_label == pred_label
    pcc_tensor = (act_label+pred_label) == 2 #買いと予測して実際どれくらいの割合が買いだったか（逆に言えば、どれくらい間違って買うか)

    return [torch.sum(acc_tensor, axis=0), [torch.sum(pcc_tensor, axis=0), torch.sum(pred_label, axis=0), torch.sum(act_label, axis=0)]]


class Early_Stopping:
    def __init__(self, stop_past, lr_past):
        self.precision = 0
        self.loss = 100000
        self.stop_past = stop_past #どれくらいの間ロスが改善しなくても見逃すか（逆に言えばこれを超えたらstop)
        self.lr_past = lr_past
        self.pre_count = 0
        self.loss_count = 0
    
    def __call__(self, precision, loss):
        if precision > self.precision:
            self.precision = precision
            self.pre_count = 0
        
        else:
            self.pre_count += 1
        
        if loss < self.loss:
            self.loss = loss
            self.loss_count = 0
        
        else:
            self.loss_count += 1
        
        if (self.pre_count > self.stop_past) and (self.loss_count > self.stop_past):
            print('Precisionおよび損失の改善が見られませんでした。')
            return True
    
    def adjust_lr(self, scheduler):
        if (self.loss_count%self.lr_past==0) and (self.loss_count!=0):
            scheduler.step()
    



def main():
    hidden_size = 64
    output_size = 2
    dropout = 0.2
    kernel_size = 5
    
    csv_path = 'd:/workSpace/virtual_coin/statics/train/jpy_rates.csv'

    ret_delta = Timedelta(hours=1) #何次リターン(のボラリティ)をイベント抽出に用いるか
    vol_term = Timedelta(days=10) #過去どれくらいの期間をもとにその時点のボラリティを計算するか
    past_term = Timedelta(days=1) #特徴量に使用する期間
    vertical_delta = Timedelta(hours=6) #垂直バリア期間
    ptSl=[1,1]

    batch_size = 32

    transform = [Standardize(), Log_scale(), FFD(d=0.4, thres=0.01), #0-2
                 Volatility(Timedelta(hours=0.5)), Volatility(Timedelta(hours=1)), #3-4
                 Volatility(Timedelta(hours=2)), Volatility(Timedelta(hours=6)), Volatility(Timedelta(days=1)),  #5-7
                 Moving_average(Timedelta(hours=0.5)), Moving_average(Timedelta(hours=1)), #8-9
                 Moving_average(Timedelta(hours=2)), Moving_average(Timedelta(hours=6)), Moving_average(Timedelta(days=1)), #10-12
                 FFD(d=0.6, thres=0.01), FFD(d=0.8, thres=0.01), FFD(d=1.0, thres=0.01)] #13-15

    max_term = past_term + Timedelta(days=1) #データを切り落とす部分。days=1は上記transformで最大termがdays=1のため
    
    feature_path = 'LSTM_model/features/feature_1d.npy'
    label_path = 'LSTM_model/labels/label_1d.npy'
    feature_load = True
    label_load = True
    feature_idx = [1]#, 2, 3, 5, 8, 10, 11] #使用する特徴量

    input_size = len(feature_idx)
    num_channels=[1, 1, 1, 1]
    
    lr = 0.01
    
    epochs_num = 500

    model_base_path = 'LSTM_model/models_ver2/' 

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device：", device)

    torch.backends.cudnn.benchmark = True

    dataset = RateDataset(csv_path, batch_size, ret_delta, vol_term, past_term, vertical_delta, max_term, feature_path, label_path, feature_idx,
                          feature_load, label_load, ptSl, transform)



    loss_weight = torch.tensor([1.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weight, reduction='mean')


    
    #テストデータの組み合わせ。10グループのうち2つをテストにする。変えないので変数にしません（というかめんどい）
    test_groups = [[i, i+j] for i in range(9) for j in range(1,10-i)]

    
    for test_group in test_groups[44: ]:

        #net = CNN_BiLSTMNet1(input_size=input_size, hidden_size=hidden_size, output_size=output_size, 
                      #dropout=dropout, kernel_size=kernel_size).to(device)
        
        net = TCN(input_size=input_size, output_size=output_size, num_channels=num_channels, 
                  kernel_size=kernel_size, dropout=dropout, seq_len=288).to(device)
        
        optimizer = Adam(net.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

        model_path = model_base_path + 'model' + str(test_group) + '.pth'
        writer = SummaryWriter(log_dir='LSTM_model/runs_ver2/'+str(test_group)+'/')
        
        best_pcc = 0

        train_dataloader, test_dataloader = dataset.makeBatch(test_group)
        train_size, test_size = dataset.get_datasize()

        early_stopping = Early_Stopping(stop_past=20, lr_past=5)
        

        for epoch in range(epochs_num):

            print('%d エポック目'%(epoch+1))

            torch.manual_seed(epoch)
            np.random.seed(epoch)

            
            label_acc_train = [0, [0, 0, 0]]

            train_loss = 0

            
            net.train()

            
            for batch in train_dataloader:
                
                train_x, train_t = batch


                #LSTMの入力は[バッチサイズ, シークエンス数, 入力サイズ]なので調整
                train_x = train_x.float().to(device)
                train_t = train_t.float().to(device)
                optimizer.zero_grad()

                
                #出力yのサイズは[バッチサイズ, 出力サイズ]となっている
                y = net(train_x)
                loss = criterion(y, train_t)

                #print('loss:', loss.item())

                loss.backward()

                optimizer.step()
                
                
              
                acc_train =  calc_accuracy(y, train_t)
                label_acc_train[0] += acc_train[0]
                label_acc_train[1][0] += acc_train[1][0]
                label_acc_train[1][1] += acc_train[1][1]
                label_acc_train[1][2] += acc_train[1][2]

    
                
                train_loss += loss

                #writer.add_scalar('Loss/train', loss, n)



            label_acc_test = [0,[0,0,0]] #正解率、precisionの順番

            test_loss = 0
            
            net.eval()
            with torch.no_grad():
                for batch in test_dataloader:

                    test_x, test_t = batch
                    test_x = test_x.float().to(device)
                    test_t = test_t.float().to(device)

                    y = net(test_x)

                    loss = criterion(y, test_t)

                    
                    acc_test = calc_accuracy(y, test_t)
                    label_acc_test[0] += acc_test[0]
                    label_acc_test[1][0] += acc_test[1][0]
                    label_acc_test[1][1] += acc_test[1][1]
                    label_acc_test[1][2] += acc_test[1][2]
        
                    
                    test_loss += loss
            
            
    
        
            
    
            #以下書き込み
            writer.add_scalar('Loss/train', train_loss.float()/train_size, epoch)
            writer.add_scalar('Loss/test', test_loss.float()/test_size, epoch)


            train_acc = label_acc_train[0].float()/train_size #単純な正解率
            train_pcc = label_acc_train[1][0].float()/label_acc_train[1][1].float() #Precision　「買い」と予測して実際あたっていた率
            train_recall = label_acc_train[1][0].float()/label_acc_train[1][2].float() #全体の「買い」のうち予測できた率
            train_f1 = 2*train_pcc*train_recall / (train_pcc+train_recall)
            

            test_acc  = label_acc_test[0].float()/test_size
            test_pcc = label_acc_test[1][0].float()/label_acc_test[1][1].float()
            test_recall = label_acc_test[1][0].float()/label_acc_test[1][2].float()
            test_f1 = 2*test_pcc*test_recall / (test_pcc+test_recall)
            test_pos_pro = label_acc_test[1][2]/test_size #テストデータの陽性割合

            if test_pcc > best_pcc and test_recall > 0.4:
                best_model = copy.deepcopy(net)
                best_pcc = test_pcc
                torch.save(best_model.state_dict(), model_path)

            #writer.add_scalar('Accuracy/train', train_acc, epoch)
            #writer.add_scalar('Accuracy/test', test_acc, epoch)
            writer.add_scalar('Precision/train', train_pcc, epoch)
            writer.add_scalar('Precision/test', test_pcc, epoch)
            writer.add_scalar('Recall/train', train_recall, epoch)
            writer.add_scalar('Recall/test', test_recall, epoch)
            writer.add_scalar('F1score/train', train_f1, epoch)
            writer.add_scalar('F1score/test', test_f1, epoch)
            
            #正解率、precision、recallを表示
            print('train_loss:%.3f, test_loss:%.3f'%(train_loss, test_loss))
            print('best_pcc:%.3f'%(best_pcc))
            print('label_acc_train:%.3f, label_acc_test:%.3f'%(train_acc, test_acc))
            print('label_pcc_train:%.3f, label_pcc_test:%.3f(陽性割合:%.3fうち)'%(train_pcc, test_pcc, test_pos_pro))
            print('label_recall_train:%.5f, label_recall_test:%.5f'%(train_recall, test_recall))


            if early_stopping(test_pcc, test_loss):
                print('学習を終了しました')
                break
            
            #改善してないなら学習率を変化させる
            early_stopping.adjust_lr(scheduler)
        
   
            



if __name__ == '__main__':
    main()