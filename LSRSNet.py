import argparse
import copy
import numpy as np
import os
import random
from time import time
from load_data import load_EOD_data, load_relation_data
from evaluator import evaluate
import torch
from torch.nn import Module, Parameter, LSTM, Linear, Conv1d, Dropout, ReLU, MaxPool1d, LeakyReLU, MSELoss, init, Softmax, GRU, RNN
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# seed = 1
# np.random.seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.manual_seed(seed) 

class StockLSTM(Module):
    def __init__(self, modeltype, input_size, lstm_hidden_size, lstm_layers, lstm_output_size, dropout_rate):
        super(StockLSTM, self).__init__()
        if modeltype == 'rnn':
            self.lstm = RNN(input_size=input_size, hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers, batch_first=True)
        if modeltype == 'lstm':
            self.lstm = LSTM(input_size=input_size, hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers, batch_first=True)
        if modeltype == 'gru':
            self.lstm = GRU(input_size=input_size, hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers, batch_first=True)
        # self.lstm = LSTM(input_size=input_size, hidden_size=lstm_hidden_size,
        #                  num_layers=lstm_layers, batch_first=True, dropout=dropout_rate)
        self.linear = Linear(in_features=lstm_hidden_size, out_features=lstm_output_size)
        self.leaky_relu = LeakyReLU(0.2)

    def forward(self, x):#, hidden=None):
        lstm_out, _ = self.lstm(x)  # [1026，16，64]
        linear_out = self.linear(lstm_out[:,-1,:])  # [1026,1]
        out = self.leaky_relu(linear_out)  # [1026,1]
        return out#, hidden

class StockMLP(Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StockMLP, self).__init__()
        self.linear1 = Linear(in_features=input_size, out_features=hidden_size)
        self.linear2 = Linear(in_features=hidden_size, out_features=output_size)
        self.leaky_relu = LeakyReLU(0.2)
    
    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], -1))
        hidden = self.linear1(x)
        output = self.linear2(hidden)
        output = self.leaky_relu(output)
        return output

class StockLSTMCNN(Module):
    def __init__(self, modeltype, input_size, lstm_hidden_size, lstm_layers, lstm_output_size, dropout_rate):
        super(StockLSTMCNN, self).__init__()
        if modeltype == 'rnn':
            self.lstm = RNN(input_size=input_size, hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers, batch_first=True)
        if modeltype == 'lstm':
            self.lstm = LSTM(input_size=input_size, hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers, batch_first=True)
        if modeltype == 'gru':
            self.lstm = GRU(input_size=input_size, hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers, batch_first=True)
        # self.lstm = LSTM(input_size=input_size, hidden_size=lstm_hidden_size,
        #                  num_layers=lstm_layers, batch_first=True, dropout=dropout_rate)
        self.conv1 = Conv1d(lstm_hidden_size, lstm_hidden_size, 3, padding=1)
        self.linear = Linear(in_features=lstm_hidden_size, out_features=lstm_output_size)
        self.leaky_relu = LeakyReLU(0.2)

    def forward(self, x):#, hidden=None):
        lstm_out, _ = self.lstm(x)  # [1026，16，64], [1737, 8, 32]
        x = lstm_out[:,-1,:].permute(1,0)
        x = torch.unsqueeze(x, dim=0)
        x = self.conv1(x)
        x = torch.squeeze(x, dim=0)
        x = x.permute(1,0)
        linear_out = self.linear(x)  # [1026,1]
        out = self.leaky_relu(linear_out)  # [1026,1]
        return out#, hidden

class Spatial_Attention_layer(Module):
    '''
    compute spatial attention scores
    '''
    def __init__(self, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = Parameter(torch.FloatTensor(num_of_timesteps))
        self.W2 = Parameter(torch.FloatTensor(in_channels, num_of_timesteps))
        self.W3 = Parameter(torch.FloatTensor(in_channels))
        self.bs = Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices))
        self.Vs = Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices))
        init.zeros_(self.bs)
        init.normal_(self.W1)
        init.normal_(self.W2)
        init.normal_(self.W3)

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T) [1, 1026, 64, 16]
        :return: (B,N,N)
        '''
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)
        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)
        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)
        S_normalized = F.softmax(S, dim=1)
        return S_normalized

class StockAtt(Module):
    def __init__(self, modeltype, input_size, lstm_hidden_size, lstm_layers, lstm_output_size, dropout_rate, seq_emd_dim, rel_dim, concat_dim, batch_size):
        super(StockAtt, self).__init__()
        # print(modeltype, str(dropout_rate))
        if modeltype == 'rnn':
            self.lstm = RNN(input_size=input_size, hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers, batch_first=True)
        if modeltype == 'lstm':
            self.lstm = LSTM(input_size=input_size, hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers, batch_first=True)
        if modeltype == 'gru':
            self.lstm = GRU(input_size=input_size, hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers, batch_first=True, bidirectional=False)
        self.dense1 = Linear(in_features=rel_dim, out_features=1)
        self.leaky_relu1 = LeakyReLU(0.2)
        self.SAt = Spatial_Attention_layer(lstm_hidden_size, batch_size, 8)
        # self.dense2 = Linear(in_features=seq_emd_dim, out_features=1)
        # self.leaky_relu2 = LeakyReLU(0.2)
        # self.dense3 = Linear(in_features=seq_emd_dim, out_features=1)
        # self.leaky_relu3 = LeakyReLU(0.2)
        self.softmax = Softmax(dim=1)
        self.dense4 = Linear(in_features=concat_dim, out_features=1, bias=False)
        self.leaky_relu4 = LeakyReLU(0.2)
        # self.all_one = torch.ones((batch_size, 1), dtype=torch.float32).cuda()
        #init.xavier_uniform_(self.lstm.weight_ih_l0)
        #init.orthogonal_(self.lstm.weight_hh_l0)
        #init.zeros_(self.lstm.bias_hh_l0)
        #init.zeros_(self.lstm.bias_ih_l0)
        init.xavier_uniform_(self.dense1.weight)
        # init.xavier_uniform_(self.dense2.weight)
        # init.xavier_uniform_(self.dense3.weight)
        init.xavier_uniform_(self.dense4.weight)

    def forward(self, x, relation, rel_mask):#, hidden=None):
        lstm_out, _ = self.lstm(x)  # [1026，16，64]
        feature = lstm_out[:,-1,:]
        # rel_weight = self.dense1(relation)
        # rel_weight = self.leaky_relu1(rel_weight)  #[1026,1026,1]
        # rel_weight = torch.unsqueeze(1.0-relation, -1)

        pvalue = relation[0,:,:]
        pearson = relation[1,:,:]
        rel_pvalue = torch.unsqueeze(pvalue, -1)
        rel_pearson = torch.unsqueeze(pearson, -1)

        # feature = torch.matmul(rel_pvalue[:,:,-1], feature) 

        Sat = self.SAt(torch.unsqueeze(lstm_out.permute(0,2,1), 0))
        # Sat = self.SAt(torch.unsqueeze(torch.unsqueeze(feature, -1), 0))

        # weight_masked = self.softmax(rel_mask*Sat[0,:,:]) #+rel_weight[:,:,-1])
        # weight_masked = self.softmax(rel_mask*(Sat[0,:,:]+rel_pearson[:,:,-1])*rel_pvalue[:,:,-1])
        weight_masked = self.softmax(rel_mask*(rel_pearson[:,:,-1]*rel_pvalue[:,:,-1]))

        # head_weight = self.dense2(feature)
        # head_weight = self.leaky_relu2(head_weight)  #[1026,1]
        # tail_weight = self.dense3(feature)
        # tail_weight = self.leaky_relu3(tail_weight)  #[1026,1]
        # weight = torch.matmul(head_weight, self.all_one.permute(1,0)) + \
        #         torch.matmul(self.all_one, tail_weight.permute(1,0)) + rel_weight[:,:,-1]
        # weight_masked = self.softmax(rel_mask + weight)  #[1026,1026]

        outputs_proped = torch.matmul(weight_masked, feature)  #[1026,64]
        # outputs_proped = feature
        outputs_concated = torch.cat([feature, outputs_proped], 1)  #[1026,128]
        # outputs_concated = feature
        predictions = self.dense4(outputs_concated)
        predictions = self.leaky_relu4(predictions)  #[1026,1]
        return predictions

class StockEndtoEnd(Module):
    def __init__(self, modeltype, input_size, lstm_hidden_size, lstm_layers, lstm_output_size, dropout_rate, seq_emd_dim, rel_dim, concat_dim, batch_size):
        super(StockEndtoEnd, self).__init__()
        print(modeltype, str(dropout_rate))
        if modeltype == 'rnn':
            self.lstm = RNN(input_size=input_size, hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers, batch_first=True)
        if modeltype == 'lstm':
            self.lstm = LSTM(input_size=input_size, hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers, batch_first=True)
        if modeltype == 'gru':
            self.lstm = GRU(input_size=input_size, hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers, batch_first=True)
        self.conv1 = Conv1d(in_channels=input_size, out_channels=lstm_hidden_size, kernel_size=8, stride=1, padding=0)
        self.conv2 = Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=3, stride=1, padding=1)
        # self.conv3 = Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=3, stride=1, padding=1, dilation=2)
        self.conv3 = Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=5, stride=1, padding=2)
        # self.linear = Linear(in_features=lstm_hidden_size, out_features=lstm_output_size)
        # self.leaky_relu = LeakyReLU(0.2)
        self.dense1 = Linear(in_features=rel_dim, out_features=1)
        self.leaky_relu1 = LeakyReLU(0.2)
        # self.dense_gcn = Linear(in_features=seq_emd_dim, out_features=lstm_output_size, bias=False)
        # self.relu = ReLU()
        self.dense2 = Linear(in_features=seq_emd_dim, out_features=1)
        # self.conv1 = Conv1d(in_channels=seq_emd_dim, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.leaky_relu2 = LeakyReLU(0.2)
        self.dense3 = Linear(in_features=seq_emd_dim, out_features=1)
        # self.conv2 = Conv1d(in_channels=seq_emd_dim, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.leaky_relu3 = LeakyReLU(0.2)
        self.softmax = Softmax(dim=0)
        self.dense4 = Linear(in_features=concat_dim, out_features=1)
        self.leaky_relu4 = LeakyReLU(0.2)
        self.all_one = torch.ones((batch_size, 1), dtype=torch.float32).cuda()
        #init.xavier_uniform_(self.lstm.weight_ih_l0)
        #init.orthogonal_(self.lstm.weight_hh_l0)
        #init.zeros_(self.lstm.bias_hh_l0)
        #init.zeros_(self.lstm.bias_ih_l0)
        init.xavier_uniform_(self.dense1.weight)
        init.xavier_uniform_(self.dense2.weight)
        init.xavier_uniform_(self.dense3.weight)
        init.xavier_uniform_(self.dense4.weight)

    def forward(self, x, relation, rel_mask):#, hidden=None):
        # conv_out2 = self.conv2(x.permute(0,2,1))
        # conv_out3 = self.conv3(x.permute(0,2,1)+conv_out2)
        # conv_out = self.conv1(x.permute(0,2,1)+conv_out2+conv_out3)
        # feature = conv_out[:,:,-1]
        lstm_out, _ = self.lstm(x)  # [1026，16，64]
        feature = lstm_out[:,-1,:]

        rel_weight = self.dense1(relation)
        rel_weight = self.leaky_relu1(rel_weight)  #[1026,1026,1]

        # rel_weight = torch.unsqueeze(relation, -1)

        # pvalue = relation[0,:,:]
        # pearson = relation[1,:,:]
        # rel_pvalue = torch.unsqueeze(pvalue, -1)
        # rel_pearson = torch.unsqueeze(pearson, -1)

        # feature_gcn = torch.matmul(rel_weight[:, :,-1], feature)
        # outputs_concated = self.dense_gcn(feature_gcn)
        # outputs_concated = self.relu(outputs_concated)

        # explict modeling
        # inner_weight = torch.matmul(feature, feature.permute(1,0))
        # weight = torch.mul(inner_weight, rel_weight[:, :, -1])

        # implict modeling
        # head_weight = self.dense2(feature)
        # head_weight = self.leaky_relu2(head_weight)  #[1026,1]
        # tail_weight = self.dense3(feature)
        # tail_weight = self.leaky_relu3(tail_weight)  #[1026,1]

        # conv modeling
        # head_weight = self.conv1(lstm_out.permute(0,2,1))
        # head_weight = head_weight[:,:,-1]
        # tail_weight = self.conv2(lstm_out.permute(0,2,1))
        # tail_weight = tail_weight[:,:,-1]
        # weight = torch.matmul(head_weight, self.all_one.permute(1,0)) + \
        #        torch.matmul(self.all_one, tail_weight.permute(1,0)) #+ rel_weight[:,:,-1]
        weight = rel_weight[:,:,-1]
        # weight = 1.0-relation
        # weight = torch.matmul(head_weight, self.all_one.permute(1,0)) + \
        #        torch.matmul(self.all_one, tail_weight.permute(1,0)) + rel_pearson[:,:,-1]
        
        weight_masked = self.softmax(rel_mask * weight)  #[1026,1026]
        # weight_masked = self.softmax((rel_mask+weight)*rel_pvalue[:,:,-1])
        
        outputs_proped = torch.matmul(weight_masked, feature)  #[1026,64]
        outputs_concated = torch.cat([feature, outputs_proped], 1)  #[1026,128]
        # outputs_concated = torch.cat([feature, feature], 1) 
        predictions = self.dense4(outputs_concated)
        # predictions = self.dense4(feature)
        predictions = self.leaky_relu4(predictions)  #[1026,1]
        return predictions

        # linear_out = self.linear(lstm_out[:,-1,:])  # [1026,1]
        # out = self.leaky_relu(linear_out)  # [1026,1]
        # return out#, hidden)

class GCNLSTM(Module):
    def __init__(self, modeltype, input_size, lstm_hidden_size, lstm_layers, lstm_output_size, dropout_rate, seq_emd_dim, rel_dim, concat_dim, batch_size):
        super(GCNLSTM, self).__init__()
        print(modeltype, str(dropout_rate))
        if modeltype == 'rnn':
            self.lstm = RNN(input_size=input_size, hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers, batch_first=True)
        if modeltype == 'lstm':
            self.lstm = LSTM(input_size=input_size, hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers, batch_first=True)
        if modeltype == 'gru':
            self.lstm = GRU(input_size=input_size, hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers, batch_first=True)
        #self.linear = Linear(in_features=lstm_hidden_size, out_features=lstm_output_size)
        #self.leaky_relu = LeakyReLU(0.2)
        self.dense1 = Linear(in_features=rel_dim, out_features=1)
        self.leaky_relu1 = LeakyReLU(0.2)
        self.dense2 = Linear(in_features=seq_emd_dim, out_features=lstm_hidden_size, bias=False)
        self.relu2 = ReLU()
        self.dense3 = Linear(in_features=lstm_hidden_size, out_features=lstm_hidden_size, bias=False)
        self.relu3 = ReLU()
        self.dense4 = Linear(in_features=concat_dim, out_features=1, bias=False)
        self.relu4 = ReLU()
        self.relu5 = LeakyReLU(0.2)
        self.softmax = Softmax(dim=0)
        self.all_one = torch.ones((batch_size, batch_size), dtype=torch.float32).cuda()
        self.SAt = Spatial_Attention_layer(lstm_hidden_size, batch_size, 8)
        init.xavier_uniform_(self.dense1.weight)
        init.xavier_uniform_(self.dense2.weight)
        init.xavier_uniform_(self.dense3.weight)
        init.xavier_uniform_(self.dense4.weight)

    def normalize_adj_torch(self, mx):
        mx = mx.to_dense()           #构建张量
        rowsum = mx.sum(1)           #每行的数加在一起
        r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()   #输出rowsum ** -1/2
        r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.         #溢出部分赋值为0
        r_mat_inv_sqrt = torch.diag(r_inv_sqrt)         #对角化
        mx = torch.matmul(mx, r_mat_inv_sqrt)
        mx = torch.transpose(mx, 0, 1)                   #转置
        mx = torch.matmul(mx, r_mat_inv_sqrt)
        return mx

    def forward(self, x, relation, rel_mask):#, hidden=None):
        lstm_out, _ = self.lstm(x)  # [1026，16，64]
        feature = lstm_out[:,-1,:]

        # rel_weight = self.dense1(relation)
        # rel_weight = self.leaky_relu1(rel_weight)  #[1026,1026,1]
        # rel_weight = torch.unsqueeze(1.0-relation, -1)
        # rel_weight[:, :,-1] = rel_weight[:, :,-1] + self.all_one
        # rel_weight[:, :,-1] = self.softmax(rel_mask + rel_weight[:, :, -1])
        Sat = self.SAt(torch.unsqueeze(lstm_out.permute(0,2,1), 0))

        pvalue = relation[0,:,:]
        pearson = relation[1,:,:]
        rel_pvalue = torch.unsqueeze(pvalue, -1)
        rel_pearson = torch.unsqueeze(pearson, -1)
        rel_weight = self.softmax((rel_mask+Sat[0,:,:]+rel_pearson[:,:,-1]))#*rel_pvalue[:,:,-1])
        # rel_weight = self.normalize_adj_torch(rel_weight)
        
        feature_gcn = torch.matmul(rel_weight, feature)
        feature_gcn = self.dense2(feature_gcn)
        feature_gcn = self.relu2(feature_gcn)
        # feature_gcn = torch.matmul(rel_weight, feature_gcn)
        # feature_gcn = self.dense3(feature_gcn)
        # feature_gcn = self.relu3(feature_gcn)
        #feature_gcn = torch.matmul(rel_weight[:, :,-1], feature_gcn)
        feature_gcn = torch.cat([feature, feature_gcn], 1)
        feature_gcn = self.dense4(feature_gcn)
        predictions = self.relu5(feature_gcn)

        # feature_gcn = torch.matmul(rel_weight[:, :,-1], feature)
        # outputs_concated = self.dense_gcn(feature_gcn)
        # outputs_concated = self.relu(outputs_concated)
        
        # weight_masked = self.softmax(rel_mask + weight)  #[1026,1026]
        # outputs_proped = torch.matmul(weight_masked, feature)  #[1026,64]

        # outputs_concated = torch.cat([feature, feature_gcn], axis=1)  #[1026,128]
        # outputs_concated = torch.matmul(rel_weight[:, :,-1], outputs_concated)
        # predictions = self.dense4(outputs_concated)
        # predictions = self.relu5(predictions)  #[1026,1]
        return predictions

class PointwiseRankLoss(Module):
    def __init__(self, alpha, batch_size):
        super(PointwiseRankLoss, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.mseloss = MSELoss(reduction='none')
        self.all_one = torch.ones((batch_size, 1), dtype=torch.float32).cuda()
        self.relu = ReLU()

    def forward(self, predictions, mask, base_prices, groundtruth):
        return_ratio = torch.div((predictions-base_prices), base_prices)
        reg_loss = self.mseloss(return_ratio, groundtruth)
        reg_loss = torch.div(torch.sum(torch.mul(reg_loss, mask)), torch.sum(mask))
        pre_pw_dif = torch.matmul(return_ratio, self.all_one.permute(1,0)) - \
            torch.matmul(self.all_one, return_ratio.permute(1,0))
        gt_pw_dif = torch.matmul(self.all_one, groundtruth.permute(1,0)) - \
            torch.matmul(groundtruth, self.all_one.permute(1,0))
        mask_pw = torch.matmul(mask, mask.transpose(1,0))
        rank_loss = torch.mean(
            self.relu(
                torch.mul(
                    torch.mul(pre_pw_dif, gt_pw_dif),
                    mask_pw
                )
            )
        )
        loss = reg_loss#+self.alpha*rank_loss
        return loss, return_ratio

class RegLoss(Module):
    def __init__(self):
        super(RegLoss, self).__init__()
        self.mseloss = MSELoss(reduction='none')

    def forward(self, predictions, mask, base_prices):
        reg_loss = self.mseloss(predictions, base_prices)
        reg_loss = torch.sum(torch.mul(reg_loss, mask))
        if reg_loss != 0:
            reg_loss = torch.div(reg_loss, torch.sum(mask))
        # reg_loss = torch.div(torch.sum(torch.mul(reg_loss, mask)), torch.sum(mask))
        return reg_loss

class RankLSTM:
    def __init__(self, modeltype, data_path, market_name, tickers_fname, relation_name, parameters,
                 steps=1, epochs=50, batch_size=None, gpu=False):
        self.data_path = data_path
        self.market_name = market_name
        self.tickers_fname = tickers_fname
        self.relation_name = relation_name
        # load data
        self.tickers = np.genfromtxt(os.path.join(data_path, '..', tickers_fname),
                                     dtype=str, delimiter='\t', skip_header=False)
        # print('#tickers selected:', len(self.tickers))

        # ground_truth, [batch_size,1], [1026,1]
        # feature, [batch_size, seq_length, feature_dim], [1026, 16, 5]
        # base_price, [batch_size,1], [1026,1]
        # all_one, [batch_size,1], [1026,1]

        self.eod_data, self.mask_data, self.gt_data, self.price_data = \
            load_EOD_data(data_path, market_name, self.tickers, steps)

        rname_tail = {'sector_industry': '_industry_relation.npy',
                      'wikidata': '_wiki_relation.npy'}

        self.rel_encoding, self.rel_mask = load_relation_data(
            os.path.join(self.data_path, '..', 'relation', self.relation_name,
                         self.market_name + rname_tail[self.relation_name])
        )
        self.rel_mask = (np.min(self.rel_mask)-self.rel_mask)/np.min(self.rel_mask)

        self.pvalue1 = np.load('/home/lby/RSR-pytorch/pvalue_NYSE.npy')
        cond1 = np.where(self.pvalue1<1, 1, 0)
        I = np.identity(1737)  #1737, 1026
        self.pvalue2 = np.load('/home/lby/RSR-pytorch/pearson_r_NYSE.npy')
        # cond2 = np.zeros([1737, 1737])
        # cond2 = np.where((self.pvalue2>-0.5)&(self.pvalue2<0.5), 0, 1)
        cond2 = np.where((self.pvalue2>-0.5)&(self.pvalue2<0.5), 0, 1)
        # self.pvalue = np.load('/home/lby/RSR-pytorch/euclidist_NYSE.npy')
        # pvalue_min = np.min(self.pvalue)
        # self.pvalue = pvalue_min/self.pvalue
        # cond = np.where(self.pvalue>0.1,1,0)
        self.pvalue = np.array([(1.0-self.pvalue1)*cond1+I, self.pvalue2*cond2])
        # print('relation encoding shape:', self.rel_encoding.shape)
        # print('relation mask shape:', self.rel_mask.shape)

        self.parameters = copy.copy(parameters)
        self.steps = steps
        self.epochs = epochs
        if batch_size is None:
            self.batch_size = len(self.tickers)
        else:
            self.batch_size = batch_size

        self.valid_index = 756
        self.test_index = 1008
        self.trade_dates = self.mask_data.shape[1]
        self.fea_dim = 5

        self.device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")
        self.model = StockAtt(modeltype, self.fea_dim, self.parameters['unit'], 1, 1, 1, seq_emd_dim=self.parameters['unit'], rel_dim=self.rel_encoding.shape[2], concat_dim=2*self.parameters['unit'], batch_size=self.batch_size)
        # self.model = StockLSTM(modeltype, self.fea_dim, self.parameters['unit'], 1, 1, 1)
        # self.model = StockLSTMCNN(modeltype, self.fea_dim, self.parameters['unit'], 1, 1, 1)
        # self.model = StockEndtoEnd(modeltype, self.fea_dim, self.parameters['unit'], 1, 1, 1, seq_emd_dim=self.parameters['unit'], rel_dim=self.rel_encoding.shape[2], concat_dim=2*self.parameters['unit'], batch_size=self.batch_size)
        # self.model = GCNLSTM(modeltype, self.fea_dim, self.parameters['unit'], 1, 1, 1, seq_emd_dim=self.parameters['unit'], rel_dim=self.rel_encoding.shape[2], concat_dim=2*self.parameters['unit'], batch_size=self.batch_size)
        # self.model = StockMLP(input_size=self.fea_dim*self.parameters['seq'], hidden_size=32, output_size=1)
        if gpu:
            self.model = self.model.cuda()

    def get_batch(self, offset=None):
        if offset is None:
            offset = random.randrange(0, self.valid_index)
        seq_len = self.parameters['seq']
        mask_batch = self.mask_data[:, offset: offset + seq_len + self.steps]
        mask_batch = np.min(mask_batch, axis=1)
        return self.eod_data[:, offset:offset + seq_len, :], \
               np.expand_dims(mask_batch, axis=1), \
               np.expand_dims(
                   self.price_data[:, offset + seq_len - 1], axis=1
               ), \
               np.expand_dims(
                   self.gt_data[:, offset + seq_len + self.steps - 1], axis=1
               )

    def get_reg_batch(self, offset=None):
        if offset is None:
            offset = random.randrange(0, self.valid_index)
        seq_len = self.parameters['seq']
        mask_batch = self.mask_data[:, offset: offset + seq_len + self.steps]
        mask_batch = np.min(mask_batch, axis=1)
        return self.eod_data[:, offset:offset + seq_len, :], \
               np.expand_dims(mask_batch, axis=1), \
               np.expand_dims(
                   self.price_data[:, offset + seq_len], axis=1
               )
               
    def train_reg(self, save_path):
        batch_offsets = np.arange(start=0, stop=self.valid_index, dtype=int)
        batch_iner_offsets = np.arange(start=0, stop=self.batch_size, dtype=int)
        best_valid_loss = np.inf
        best_epoch = 0
        # best_valid_perf = 0
        best_test_mse = 0
        best_test_mae = 0
        best_test_mape = 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.parameters['lr'])
        criterion = RegLoss()
        train_pvalue = torch.from_numpy(self.pvalue).to(self.device).float()
        train_relation = torch.from_numpy(self.rel_encoding).to(self.device).float()
        ave_test_time = 0
        best_test_predictions = []
        for epoch in range(self.epochs):
            # print("Epoch {}/{}".format(epoch, self.epochs))
            self.model.train()
            train_loss = []
            np.random.shuffle(batch_offsets)
            # for i, _data in enumerate(train_loader):
            for i in range(self.valid_index-self.parameters['seq']-self.steps+1):
                eod_batch, mask_batch, price_batch = self.get_reg_batch(batch_offsets[i])
                train_eod = torch.from_numpy(eod_batch).to(self.device)
                train_mask = torch.from_numpy(mask_batch).to(self.device)
                train_price = torch.from_numpy(price_batch).to(self.device)
                # train_gt = torch.from_numpy(gt_batch).to(self.device)

                # _train_X, _train_Y = _data[0].to(self.device),_data[1].to(self.device)
                optimizer.zero_grad()
                # np.random.shuffle(batch_iner_offsets)
                # train_eod = train_eod[batch_iner_offsets]
                # train_mask = train_mask[batch_iner_offsets]
                # train_price = train_price[batch_iner_offsets]
                # predictions = self.model(train_eod)
                # predictions = self.model(train_eod, train_relation, train_mask)
                predictions = self.model(train_eod, train_pvalue, train_mask)
                loss = criterion(predictions, train_mask, train_price)
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()
            ### evaluate
            self.model.eval()
            valis_loss = []
            # cur_valid_pred = np.zeros((len(self.tickers), self.test_index-self.valid_index), dtype=float)
            # cur_valid_gt = np.zeros((len(self.tickers), self.test_index-self.valid_index), dtype=float)
            # cur_valid_mask = np.zeros((len(self.tickers), self.test_index-self.valid_index), dtype=float)
            for cur_offset in range(
                self.valid_index - self.parameters['seq'] - self.steps + 1,
                self.test_index - self.parameters['seq'] - self.steps + 1
            ):
                eod_batch, mask_batch, price_batch = self.get_reg_batch(
                    cur_offset)
                valid_eod = torch.from_numpy(eod_batch).to(self.device)
                valid_mask = torch.from_numpy(mask_batch).to(self.device)
                valid_price = torch.from_numpy(price_batch).to(self.device)
                # valid_gt = torch.from_numpy(gt_batch).to(self.device)
            # for _valid_X, _valid_Y in valid_loader:
            #     _valid_X, _valid_Y = _valid_X.to(self.device), _valid_Y.to(self.device)
                # predictions = self.model(valid_eod, train_relation, valid_mask)
                predictions = self.model(valid_eod, train_pvalue, valid_mask)
                # predictions = self.model(valid_eod)
                loss = criterion(predictions, valid_mask, valid_price)
                valis_loss.append(loss.item())
                # cur_valid_pred[:, cur_offset-(self.valid_index-self.parameters['seq']-self.steps+1)] = cur_rr.cpu().detach().numpy()[:, 0]
                # cur_valid_gt[:, cur_offset-(self.valid_index-self.parameters['seq']-self.steps+1)] = gt_batch[:, 0]
                # cur_valid_mask[:, cur_offset-(self.valid_index-self.parameters['seq']-self.steps+1)] = mask_batch[:, 0]
            train_loss_cur = np.mean(train_loss)
            valid_loss_cur = np.mean(valis_loss)
            # print("The train loss is {:.6f}. ".format(train_loss_cur) + "The valid loss is {:.6f}.".format(valid_loss_cur))
            # cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)
            #print('\t Valid preformance:', cur_valid_perf)
            ### test
            test_loss = []
            test_mape = []
            test_mae = []
            test_predictions = []
            # cur_test_pred = np.zeros((len(self.tickers), self.trade_dates-self.test_index), dtype=float)
            # cur_test_gt = np.zeros((len(self.tickers), self.trade_dates-self.test_index), dtype=float)
            # cur_test_mask = np.zeros((len(self.tickers), self.trade_dates-self.test_index), dtype=float)
            for cur_offset in range(
                self.test_index - self.parameters['seq'] - self.steps + 1,
                self.trade_dates - self.parameters['seq'] - self.steps + 1
            ):
                eod_batch, mask_batch, price_batch = self.get_reg_batch(
                    cur_offset)
                test_eod = torch.from_numpy(eod_batch).to(self.device)
                test_mask = torch.from_numpy(mask_batch).to(self.device)
                test_price = torch.from_numpy(price_batch).to(self.device)
                # test_gt = torch.from_numpy(gt_batch).to(self.device)
                since = time()
                # predictions = self.model(test_eod, train_relation, test_mask)
                predictions = self.model(test_eod, train_pvalue, test_mask)
                # predictions = self.model(test_eod)
                # if test_predictions == []:
                #     test_predictions = predictions.cpu().detach().numpy()
                # else:
                #     test_predictions = np.concatenate((test_predictions, predictions.cpu().detach().numpy()), axis=1)
                # test_predictions.append(predictions.cpu().detach().numpy())
                ave_test_time += time()-since
                loss = criterion(predictions, test_mask, test_price)
                test_loss.append(loss.item())
                mae = abs(predictions - test_price)
                mae = torch.div(torch.sum(torch.mul(mae, test_mask)), torch.sum(test_mask))
                test_mae.append(mae.item())
                mape = abs((predictions - test_price)/test_price)
                mape = torch.div(torch.sum(torch.mul(mape, test_mask)), torch.sum(test_mask))
                test_mape.append(mape.item())
            # print('time:', ave_test_time/(epoch+1))
                # cur_test_pred[:, cur_offset-(self.test_index-self.parameters['seq']-self.steps+1)] = cur_rr.cpu().detach().numpy()[:, 0]
                # cur_test_gt[:, cur_offset-(self.test_index-self.parameters['seq']-self.steps+1)] = gt_batch[:, 0]
                # cur_test_mask[:, cur_offset-(self.test_index-self.parameters['seq']-self.steps+1)] = mask_batch[:, 0]
            test_loss_cur = np.mean(test_loss)
            test_mae_cur = np.mean(test_mae)
            test_mape_cur = np.mean(test_mape)

            # print("The test loss is {:.6f}. ".format(test_loss_cur))
            # cur_test_perf = evaluate(cur_test_pred, cur_test_gt, cur_test_mask)
            #print('\t Test performance:', cur_test_perf)

            if valid_loss_cur < best_valid_loss:
                best_epoch = epoch
                best_valid_loss = valid_loss_cur
                # best_valid_perf = cur_valid_perf
                best_test_mse = test_loss_cur
                best_test_mae = test_mae_cur
                best_test_mape = test_mape_cur
                best_test_predictions = test_predictions
                # torch.save(self.model.state_dict(), save_path)
        ave_test_time = ave_test_time/self.epochs
        # np.save('/home/lby/RSR-pytorch/data/predictions/NYSE_sat_predictions.npy', best_test_predictions)
        # torch.save(best_model.state_dict(), save_path)
        print('\nInference Time: ', ave_test_time)
        print('\nBest Epoch: ', best_epoch)
        print('\nBest Valid MSE:', best_valid_loss)
        print('Best Test MSE:', best_test_mse)
        print('Best Test MAE:', best_test_mae)
        print('Best Test MAPE:', best_test_mape)

    def train(self, save_path):
        # train_loader = DataLoader(TensorDataset(self.train_eod, self.train_gt), batch_size=self.batch_size)
        # valid_loader = DataLoader(TensorDataset(self.valid_eod, self.valid_gt), batch_size=self.batch_size)
        batch_offsets = np.arange(start=0, stop=self.valid_index, dtype=int)
        best_valid_loss = np.inf
        best_epoch = 0
        best_valid_perf = {
            'mse': np.inf, 'top1': 0.0, 'top5': 0.0, 'top10': 0.0, 'mrrt': 0.0,
            'btl': 0.0, 'abtl': 0.0, 'btl5': 0.0, 'abtl5': 0.0, 'btl10': 0.0,
            'abtl10': 0.0, 'rho': -1.0
        }
        best_test_perf = {
            'mse': np.inf, 'top1': 0.0, 'top5': 0.0, 'top10': 0.0, 'mrrt': 0.0,
            'btl': 0.0, 'abtl': 0.0, 'btl5': 0.0, 'abtl5': 0.0, 'btl10': 0.0,
            'abtl10': 0.0, 'rho': -1.0
        }

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.parameters['lr'])
        criterion = PointwiseRankLoss(self.parameters['alpha'], self.batch_size).to(self.device)
        train_relation = torch.from_numpy(self.rel_encoding).to(self.device).float()
        ave_test_time = 0
        for epoch in range(self.epochs):
            #print("Epoch {}/{}".format(epoch, self.epochs))
            self.model.train()
            train_loss = []
            np.random.shuffle(batch_offsets)
            # for i, _data in enumerate(train_loader):
            for i in range(self.valid_index-self.parameters['seq']-self.steps+1):
                eod_batch, mask_batch, price_batch, gt_batch = self.get_batch(batch_offsets[i])
                train_eod = torch.from_numpy(eod_batch).to(self.device)
                train_mask = torch.from_numpy(mask_batch).to(self.device)
                train_price = torch.from_numpy(price_batch).to(self.device)
                train_gt = torch.from_numpy(gt_batch).to(self.device)

                # _train_X, _train_Y = _data[0].to(self.device),_data[1].to(self.device)
                optimizer.zero_grad()
                predictions = self.model(train_eod, train_relation, train_mask)
                loss, _ = criterion(predictions, train_mask, train_price, train_gt)
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()
            ### evaluate
            self.model.eval()
            valis_loss = []
            cur_valid_pred = np.zeros((len(self.tickers), self.test_index-self.valid_index), dtype=float)
            cur_valid_gt = np.zeros((len(self.tickers), self.test_index-self.valid_index), dtype=float)
            cur_valid_mask = np.zeros((len(self.tickers), self.test_index-self.valid_index), dtype=float)
            for cur_offset in range(
                self.valid_index - self.parameters['seq'] - self.steps + 1,
                self.test_index - self.parameters['seq'] - self.steps + 1
            ):
                eod_batch, mask_batch, price_batch, gt_batch = self.get_batch(
                    cur_offset)
                valid_eod = torch.from_numpy(eod_batch).to(self.device)
                valid_mask = torch.from_numpy(mask_batch).to(self.device)
                valid_price = torch.from_numpy(price_batch).to(self.device)
                valid_gt = torch.from_numpy(gt_batch).to(self.device)
            # for _valid_X, _valid_Y in valid_loader:
            #     _valid_X, _valid_Y = _valid_X.to(self.device), _valid_Y.to(self.device)
                predictions = self.model(valid_eod, train_relation, valid_mask)
                loss, cur_rr = criterion(predictions, valid_mask, valid_price, valid_gt)
                valis_loss.append(loss.item())
                cur_valid_pred[:, cur_offset-(self.valid_index-self.parameters['seq']-self.steps+1)] = cur_rr.cpu().detach().numpy()[:, 0]
                cur_valid_gt[:, cur_offset-(self.valid_index-self.parameters['seq']-self.steps+1)] = gt_batch[:, 0]
                cur_valid_mask[:, cur_offset-(self.valid_index-self.parameters['seq']-self.steps+1)] = mask_batch[:, 0]
            train_loss_cur = np.mean(train_loss)
            valid_loss_cur = np.mean(valis_loss)
            #print("The train loss is {:.6f}. ".format(train_loss_cur) + "The valid loss is {:.6f}.".format(valid_loss_cur))
            cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)
            #print('\t Valid preformance:', cur_valid_perf)
            ### test
            test_loss = []
            cur_test_pred = np.zeros((len(self.tickers), self.trade_dates-self.test_index), dtype=float)
            cur_test_gt = np.zeros((len(self.tickers), self.trade_dates-self.test_index), dtype=float)
            cur_test_mask = np.zeros((len(self.tickers), self.trade_dates-self.test_index), dtype=float)
            test_time = 0
            for cur_offset in range(
                self.test_index - self.parameters['seq'] - self.steps + 1,
                self.trade_dates - self.parameters['seq'] - self.steps + 1
            ):
                eod_batch, mask_batch, price_batch, gt_batch = self.get_batch(
                    cur_offset)
                test_eod = torch.from_numpy(eod_batch).to(self.device)
                test_mask = torch.from_numpy(mask_batch).to(self.device)
                test_price = torch.from_numpy(price_batch).to(self.device)
                test_gt = torch.from_numpy(gt_batch).to(self.device)
                since = time()
                predictions = self.model(test_eod, train_relation, test_mask)
                test_time += time() -since
                loss, cur_rr = criterion(predictions, test_mask, test_price, test_gt)
                test_loss.append(loss.item())
                cur_test_pred[:, cur_offset-(self.test_index-self.parameters['seq']-self.steps+1)] = cur_rr.cpu().detach().numpy()[:, 0]
                cur_test_gt[:, cur_offset-(self.test_index-self.parameters['seq']-self.steps+1)] = gt_batch[:, 0]
                cur_test_mask[:, cur_offset-(self.test_index-self.parameters['seq']-self.steps+1)] = mask_batch[:, 0]
            test_loss_cur = np.mean(test_loss)
            ave_test_time += test_time
            #print("The test loss is {:.6f}. ".format(test_loss_cur))
            cur_test_perf = evaluate(cur_test_pred, cur_test_gt, cur_test_mask)
            #print('\t Test performance:', cur_test_perf)
            if valid_loss_cur < best_valid_loss:
                best_epoch = epoch
                best_valid_loss = valid_loss_cur
                best_valid_perf = cur_valid_perf
                best_test_perf = cur_test_perf
                torch.save(self.model.state_dict(), save_path)
        ave_test_time /= self.epochs
        print("\nTest time: {:.6f}".format(ave_test_time))
        print('\nBest Epoch:', best_epoch)
        print('\nBest Valid performance:', best_valid_perf)
        print('\tBest Test performance:', best_test_perf)

    def eval(self, ckp_path):
        self.model.load_state_dict(torch.load(ckp_path))
        self.model.eval()
        test_time = 0
        for cur_offset in range(
            self.test_index - self.parameters['seq'] - self.steps + 1,
            self.trade_dates - self.parameters['seq'] - self.steps + 1
        ):
            eod_batch, mask_batch, price_batch, gt_batch = self.get_batch(
                cur_offset)
            test_eod = torch.from_numpy(eod_batch).to(self.device)
            test_mask = torch.from_numpy(mask_batch).to(self.device)
            test_price = torch.from_numpy(price_batch).to(self.device)
            test_gt = torch.from_numpy(gt_batch).to(self.device)
            since = time()
            predictions = self.model(test_eod, train_relation, test_mask)
            test_time += time()-since
            loss, cur_rr = criterion(predictions, test_mask, test_price, test_gt)
            test_loss.append(loss.item())
            cur_test_pred[:, cur_offset-(self.test_index-self.parameters['seq']-self.steps+1)] = cur_rr.cpu().detach().numpy()[:, 0]
            cur_test_gt[:, cur_offset-(self.test_index-self.parameters['seq']-self.steps+1)] = gt_batch[:, 0]
            cur_test_mask[:, cur_offset-(self.test_index-self.parameters['seq']-self.steps+1)] = mask_batch[:, 0]
        test_loss_cur = np.mean(test_loss)
        #print("The test loss is {:.6f}. ".format(test_loss_cur))
        cur_test_perf = evaluate(cur_test_pred, cur_test_gt, cur_test_mask)
        # test_time = test_time/((self.trade_dates-self.test_index)*eod_batch.shape[0])
        print("Test time is {:.6f}. ".format(test_time))
        print("The test loss is {:.6f}. ".format(test_loss_cur))
        print('\t Test performance:', cur_test_perf)

if __name__ == '__main__':
    desc = 'train a rank lstm model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-p', help='path of EOD data',
                        default='./data/2013-01-01')
    parser.add_argument('-sp', help='path of EOD data',
                        default='./weights/model_sat_mask.pth')
    parser.add_argument('-m', help='market name', default='NYSE')
    parser.add_argument('-t', help='fname for selected tickers')
    parser.add_argument('-l', default=8,
                        help='length of historical sequence for feature')
    parser.add_argument('-u', default=32,
                        help='number of hidden units in lstm')
    parser.add_argument('-s', default=1,
                        help='steps to make prediction')
    parser.add_argument('-r', default=0.001,
                        help='learning rate')
    parser.add_argument('-a', default=10,
                        help='alpha, the weight of ranking loss')
    parser.add_argument('-g', '--gpu', type=int, default=1, help='use gpu')
    parser.add_argument('-type', type=str, default='lstm', help='type of recurrent unit')
    parser.add_argument('-e', '--emb_file', type=str,
                        default='NYSE_rank_lstm_seq-8_unit-32_0.csv.npy',
                        help='fname for pretrained sequential embedding')
    parser.add_argument('-rn', '--rel_name', type=str,
                        default='wikidata',
                        help='relation type: sector_industry or wikidata')
    args = parser.parse_args()

    if args.t is None:
        args.t = args.m + '_tickers_qualify_dr-0.98_min-5_smooth.csv'
    args.gpu = (args.gpu == 1)

    parameters = {'seq': int(args.l), 'unit': int(args.u), 'lr': float(args.r),
                  'alpha': float(args.a)}
    # print('arguments:', args)
    # print('no seed, init: xavier')
    print('parameters:', parameters)

    rank_LSTM = RankLSTM(
        modeltype=args.type,
        data_path=args.p,
        market_name=args.m,
        tickers_fname=args.t,
        parameters=parameters,
        steps=1, epochs=90, batch_size=None, gpu=args.gpu,
        relation_name=args.rel_name
    )
    rank_LSTM.train_reg(save_path=args.sp)
    # rank_LSTM.train_reg()
