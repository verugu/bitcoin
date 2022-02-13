#LSTMで出力を、その先で上昇するなら0-1の確率（つまり底）。下降するなら-1~0の確率（つまり山）とする、というアイデア。
#問題はラベル付け。上昇する、下降する、という事象をどう定量化された指数に落とし込むか。
#とりあえず下降の方はいい。大事なのは買う時なので上昇の方をどうするか。

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

class RateDataset(torch.utils.data.Dataset):
    #past_numは過去何点で予測するか。predict_numはその先何点からret_max, risk_maxを算出するか
    def __init__(self, csv_path, past_num, predict_num,train_prop, batch_size, model_mode='num_predict', transform=None):
        self.rate_data = np.array(pd.read_csv(csv_path).loc[:, 'rate'])
        self.data_num = self.rate_data.shape[0]
        self.past_num = past_num
        self.predict_num = predict_num
        self.train_prop = train_prop
        self.batch_size = batch_size
        self.model_mode = model_mode
        self.transform = transform

        self.len = self.data_num-self.past_num-self.predict_num+1
        

        dataset = []
        for idx in range(self.data_num-self.past_num-self.predict_num+1):
            #データセット作るのに必要な範囲をまとめて抽出（ある地点からpast_num遡ったデータ群と、ret,riskのために必要な先のデータ群)
            rate_data = self.rate_data[idx : idx+self.past_num+self.predict_num]
            

            #StandardizeとLog_scale２つを特徴量にしてみたかった。
            if self.transform:
                inputs = np.zeros([self.past_num+self.predict_num, len(self.transform)])

                for i, transform in enumerate(self.transform):
                #この時点でrate_dataがself.rate_dataから抽出したものであることによるindexの違いに注意
                    rate_data_temp = transform(rate_data, self.past_num-1)
                    inputs[:rate_data_temp.shape[0], i] = rate_data_temp
            
            rate_train = inputs[ : self.past_num, :]
            #インデックスの終点を書いているのは、念のため24点から計算していることを保証したいから。（範囲外ならエラーくるはず）
            #この指標はStandardizeがもとで、原則transformにはStandardizeから書くようにして、0列目がStandardizeのデータになるようにする前提

            ret_risk_label = [np.nanmax(inputs[self.past_num : self.past_num+self.predict_num, 0]), np.nanmin(inputs[self.past_num : self.past_num+self.predict_num, 0])]

            if self.model_mode == 'label_predict':
                if ret_risk_label[0] >= 0.3 and ret_risk_label[1] >= -0.1:
                    ret_risk_label = [1.0, 0.0]
                else:
                    ret_risk_label = [0.0, 1.0]
            
            #配列の大きさ違うのでタプル型
            dataset.append((torch.tensor(rate_train), torch.tensor(ret_risk_label)))
        
        self.dataset = dataset
    
    #こうしとけばデータの範囲があらかじめ指定できるはず
    def __len__(self):
        return (self.len)
    
    def __getitem__(self, idx):
        #データセット作るのに必要な範囲をまとめて抽出（ある地点からpast_num遡ったデータ群と、ret,riskのために必要な先のデータ群)
        return self.dataset[idx]
    
    def makeBatch(self):
        train_size = int(self.len*self.train_prop)
        indices = np.arange(self.len)
        #train_set, test_set = torch.utils.data.random_split(self, [train_size, test_size])

        #ある地点より過去のデータのみで学習し、testはその後のデータを用いてやるために、ランダムにはしない
        train_set = torch.utils.data.Subset(self.dataset, indices[:train_size])
        test_set = torch.utils.data.Subset(self.dataset, indices[train_size:])
        

        train_dataloader = torch.utils.data.DataLoader(train_set, self.batch_size, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_set, self.batch_size, shuffle=False)

        return train_dataloader, test_dataloader
    
    def get_datasize(self):
        train_size = int(self.len*self.train_prop)
        test_size = self.len - train_size

        return train_size, test_size


#不均衡解消。dataloaderと同じイテレータで扱うようにして、なるべくmainのコードが変わらないように。
class Balanced_dataloader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.features = np.array([feature.numpy() for feature, _ in self.dataset])
        self.labels = np.array([label.numpy() for _,label in self.dataset])

        labels = [label.numpy()[0] for _,label in dataset]
        self.major_idxs = np.where(labels == np.float32(0))[0]
        self.minor_idxs = np.where(labels == np.float32(1))[0]
        
        np.random.shuffle(self.major_idxs)
        np.random.shuffle(self.minor_idxs)

        print(self.major_idxs)

        self.batch_size = batch_size

        self.used_idx = 0
        self.count = 0
    
    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            idxs = self.major_idxs[self.used_idx : self.used_idx + self.batch_size//2].tolist()\
                   + np.random.choice(self.minor_idxs, self.batch_size//2, replace=False).tolist()
            
            yield torch.tensor(self.features[idxs]), torch.tensor(self.labels[idxs])

            self.used_idx += self.batch_size//2
            self.count += self.batch_size
        


#なんとなく親クラスとして作ってみた。最大最小正規化を外側に書きたくなかっただけ。
class Transform(object):
    def __init__(self):
        pass
    
    def __call__(self, rate_data, sd_idx):
        pass
    
    def std_max_min(self, array):
        max = np.nanmax(array)
        min = np.nanmin(array)
        return (array - min) / (max - min)

#利益率
class Standardize(Transform):
    def __init__(self):
        super().__init__()
    
    #rate_dataのどのインデックス要素を使って標準化するか
    def __call__(self, rate_data, sd_idx):
        rate = copy.deepcopy(rate_data)
        rate[sd_idx+1 : ] = (rate_data[sd_idx+1 : ] - rate_data[sd_idx])*100 / rate_data[sd_idx]
        rate[: sd_idx+1] = super().std_max_min(rate[: sd_idx+1])
        return rate

#log変換
class Log_scale(Transform):
    def __init__(self):
        super().__init__()

    def __call__(self, rate_data, sd_idx):
        log_data = np.log(rate_data)
        log_data = super().std_max_min(log_data)
        return log_data

#logの階差
class Log_sub(Transform):
    def __init__(self):
        super().__init__()

    def __call__(self, rate_data, sd_idx):
        log_data = np.log(rate_data)
        log_sub_data = super().std_max_min(np.diff(log_data, n=1, prepend=log_data[0]))
        return log_sub_data

#移動平均
class Moving_average(Transform):
    def __init__(self, n_window):
        self.n_window = n_window
    
    def __call__(self, rate_data, sd_idx):
        #paddingすることで移動平均した配列をpast_numと同じ大きさ（n_windowが偶数なら1大きい）の配列にしている
        move_average = np.pad(rate_data[: sd_idx+1], self.n_window//2, 'reflect')
        move_average = np.convolve(move_average, np.ones(self.n_window), mode='valid')/self.n_window
        move_average = super().std_max_min(move_average)
        return move_average
        


#モデル定義
class LSTMNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout, model_mode, batch_first=True):
        super(LSTMNet, self).__init__()
        self.model_mode = model_mode
        self.rnn = nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            batch_first = batch_first)
        self.drop = nn.Dropout(dropout)
        #self.linear1 = nn.Linear(hidden_size, hidden_size)
        #self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

        nn.init.xavier_normal_(self.rnn.weight_ih_l0)
        nn.init.orthogonal_(self.rnn.weight_hh_l0)

    def forward(self, inputs):
        h, _= self.rnn(inputs)
        h = self.drop(h)
        output = self.linear3(h[:, -1, :])
        
        if self.training == False:
            output = torch.softmax(output, dim=1,)
        #output = self.linear2(output)
        #output = self.linear3(output)

        return output

#ロス関数。クロスエントロピーについて、1を買いとする。0なのに1と予測には大きなロスを、１なのに０と予測には小さなロスを割り当てたい。




def calc_accuracy(predict, label, mu, error, mode='match'):
    
    #一致率
    if mode == 'match':
        acc_tensor = abs((predict-label)/label) <= mu
        
    #符号一致率
    elif mode == 'sign':
        acc_tensor = predict*label > 0
    
    elif mode == 'error':
        acc_tensor = abs(predict-label) <= error
    
    elif mode == 'label':
        #0が買い、1が売り、2が何もしない
        act_label = torch.min(label.data, 1)[1]
        pred_label = torch.min(predict.data, 1)[1]
        acc_tensor = act_label == pred_label
        pcc_tensor = (act_label+pred_label) == 2 #買いと予測して実際どれくらいの割合が買いだったか（逆に言えば、どれくらい間違って買うか)

        return [torch.sum(acc_tensor, axis=0), [torch.sum(pcc_tensor, axis=0), torch.sum(pred_label, axis=0), torch.sum(act_label, axis=0)]]
    
    return torch.sum(acc_tensor, axis=0)



def main():
    input_size = 5
    hidden_size = 128
    output_size = 2
    dropout = 0.2 

    past_num = 576 #過去どれくらいのデータを使うか。データは5分で1個。
    predict_num = 24 #2時間分
    n_window = [7, 19] #移動平均に用いる窓幅
    csv_path = 'd:/workSpace/virtual_coin/statics/train/rates.csv'
    train_prop = 0.8
    batch_size = 128
    transform = [Standardize(), Log_scale(), Log_sub(), Moving_average(n_window[0]), Moving_average(n_window[1])]
    
    #モデルを切り替え可能にする。numはreturn_max,risk_maxを予測、labelは買いを１、何もしないを０とした場合の２値分類予測を行う。
    model_modes = ['num_predict', 'label_predict']
    model_mode = 'label_predict'
    
    
    lr = 0.01
    epochs_num = 300

    mu = 0.5 #正解をどの範囲までにするか
    error = 0.2 #絶対誤差の許容範囲。単位は%

    modes = ['match', 'sign', 'error', 'label']
    mode = 'label'

    model_path = 'LSTM_model/models/label576_weight40_60.pth' 



    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device：", device)

    dataset = RateDataset(csv_path=csv_path, past_num=past_num, predict_num=predict_num, train_prop=train_prop, batch_size=batch_size, model_mode=model_mode, transform=transform)
    train_size, test_size = dataset.get_datasize()

    
    net = LSTMNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size, dropout=dropout, model_mode=model_mode).to(device)
    

    loss_weight = torch.tensor([0.4, 0.6]).to(device)

    criterion = nn.MSELoss(reduction='mean')
    if model_mode == 'label_predict':
        criterion = nn.CrossEntropyLoss(weight=loss_weight)
    optimizer = SGD(net.parameters(), lr=lr)

    writer = SummaryWriter(log_dir='runs2/')

    
    
    for epoch in range(epochs_num):

        
        torch.manual_seed(epoch)
        np.random.seed(epoch)
        
        if model_mode == 'label_predict':
            #訓練データは不均衡データを矯正
            train_dataloader = Balanced_dataloader(dataset.dataset[:train_size], batch_size)
            _, test_dataloader = dataset.makeBatch()

        else:
            train_dataloader, test_dataloader = dataset.makeBatch()
        

        train_ret_accuracy = 0
        train_risk_accuracy = 0
        
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
            
            if model_mode == 'label_predict':
                acc_train =  calc_accuracy(y, train_t, mu, error, mode=mode)
                label_acc_train[0] += acc_train[0]
                label_acc_train[1][0] += acc_train[1][0]
                label_acc_train[1][1] += acc_train[1][1]
                label_acc_train[1][2] += acc_train[1][2]

            else:
                train_acc = calc_accuracy(y, train_t, mu, error, mode=mode)
                train_ret_accuracy += train_acc[0]
                train_risk_accuracy += train_acc[1]
            
            train_loss += loss

            #writer.add_scalar('Loss/train', loss, n)


        
        
        test_ret_accuracy = 0
        test_risk_accuracy = 0

        label_acc_test = [0,[0,0,0]] #正解率、precisionの順番

        test_loss = 0
        best_loss = 100000
        
        net.eval()
        with torch.no_grad():
            for batch in test_dataloader:

                test_x, test_t = batch
                test_x = test_x.float().to(device)
                test_t = test_t.float().to(device)

                y = net(test_x)

                loss = criterion(y, test_t)

                
                if model_mode == 'label_predict':
                    acc_test = calc_accuracy(y, test_t, mu, error, mode=mode)
                    label_acc_test[0] += acc_test[0]
                    label_acc_test[1][0] += acc_test[1][0]
                    label_acc_test[1][1] += acc_test[1][1]
                    label_acc_test[1][2] += acc_test[1][2]
                
                else:
                    test_acc = calc_accuracy(y, test_t, mu, error, mode=mode)
                    test_ret_accuracy += test_acc[0]
                    test_risk_accuracy += test_acc[1]
                
                test_loss += loss
        
        
        if test_loss < best_loss:
            best_model = copy.deepcopy(net)
            best_loss = test_loss
            torch.save(best_model.state_dict(), model_path)
        





        writer.add_scalar('Loss/train', train_loss.float()/train_size, epoch)
        writer.add_scalar('Loss/test', test_loss.float()/test_size, epoch)

        #writer.add_scalar('Loss/train_ret', train_ret_accurancy, epoch)
        #writer.add_scalar('Loss/train_risk', train_risk_accurancy, epoch)
        #writer.add_scalar('Loss/test_ret', test_ret_accurancy, epoch)
        #writer.add_scalar('Loss/test_risk', test_risk_accurancy, epoch)

        print('%d エポック目'%(epoch+1))

        if model_mode == 'label_predict':
            train_acc = label_acc_train[0].float()/train_size
            train_pcc = label_acc_train[1][0].float()/label_acc_train[1][1].float()
            train_recall = label_acc_train[1][0].float()/label_acc_train[1][2].float()

            test_acc  = label_acc_test[0].float()/test_size
            test_pcc = label_acc_test[1][0].float()/label_acc_test[1][1].float()
            test_recall = label_acc_test[1][0].float()/label_acc_test[1][2].float()

            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/test', test_acc, epoch)
            writer.add_scalar('Precision/train', train_pcc, epoch)
            writer.add_scalar('Precision/test', test_pcc, epoch)

            print('label_acc_train:%.3f, label_acc_test:%.3f'%(train_acc, test_acc))
            print('label_pcc_train:%.3f, label_pcc_test:%.3f'%(train_pcc, test_pcc))


            print('label_recall_train:%.5f, label_recall_test:%.5f'%(train_recall, test_recall))
        else:
            train_ret_accuracy = train_ret_accuracy.float()/train_size
            train_risk_accuracy = train_risk_accuracy.float()/train_size
            test_ret_accuracy = test_ret_accuracy.float()/test_size
            test_risk_accuracy = test_risk_accuracy.float()/test_size

            print('train_ret_acc:%.3f, train_risk_acc:%.3f'%(train_ret_accuracy, train_risk_accuracy))
            print('test_ret_acc:%.3f, test_risk_acc:%.3f'%(test_ret_accuracy, test_risk_accuracy)) 
        
        print('train_loss:%.3f'%(train_loss))

        writer.close()
    


    


if __name__ == '__main__':
    main()


