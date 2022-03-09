import multiprocessing
import numpy as np
import pandas as pd


class multi:
    def __init__(self):
        t = pd.DatetimeIndex(np.arange(10, dtype='datetime64[s]'))
        with multiprocessing.Pool(processes=6) as pool:
            features_temp = pool.map(self.make_features, t)
    
        self.feature = features_temp       

    def make_features(self, time):
        return time+pd.Timedelta(days=1)


def main():
    features_temp = multi()
    print(features_temp.feature)

if __name__ == '__main__':

    #現在までの必要なデータを作成する取ってくる。timestampとレート。結局jp→utc変換が必要か。
    
    #LSTMモデルを読み込む。これは2020, 2021年のデータをもとに事前訓練しておく。
    
    #ループ開始。5分おきでループ。以下ループ内
    
    #LSTMに必要な現在までの5分おきデータを取得。↑に追加する形にするか、ここで一括でまとめるかは後で。
    
    #privateapi認証

    #自分の注文を見る。とりあえず１注文で動かすので、何もないなら買いの判断へ。btc持ってるなら売りの判断へ。

    #現在の板のスプレッドを見て、一番安い売り、一番高い買いを見る。

    #if 買いの判断。lstm出力による。買う場合は同時に高めで売りの注文を飛ばしても良いかも。(5分の間の急騰に備える)

    #else　売りの判断。同様にlstm。↑で売りの注文出してるならこれをキャンセルするか、続行しないといけない。

    #order種類、数量を指定して売り買いを行う。

    #slackに通知。内容は売り買い注文した数量（ここでは約定してないことに注意)、レート。売った場合はそれを買ったときのレートから利益を算出。
    #過去数件の約定の内容。

    #以上、一番大変なのはlstmモデルですね...。
    
    main()


    

