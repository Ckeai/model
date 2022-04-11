from __future__ import print_function, division
import argparse
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear



class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__() ##super,额外引用nn.module的初始化
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)
        
        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)


    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, z

# def target_distribution(q):###？？？？？？？？
#     weight = q ** 2 / q.sum(0)
#     return (weight.t() / weight.sum(1)).t()
#
# class IDEC(nn.Module): #增加了P,Q自监督后的AE模型，P通过降维特征和kmeans的聚类中心计算聚类得到，P表示节点相对所有类别的得分
#     def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
#                  n_input, n_z, n_clusters, v=1):
#         super(IDEC, self).__init__()
#         self.ae = AE(
#             n_enc_1=n_enc_1,
#             n_enc_2=n_enc_2,
#             n_enc_3=n_enc_3,
#             n_dec_1=n_dec_1,
#             n_dec_2=n_dec_2,
#             n_dec_3=n_dec_3,
#             n_input=n_input,
#             n_z=n_z)
#         #self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))
#
#
#         self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z)) #帮助使CLUSTER_LAYER作为parameter参数，可在训练中进行学习
#         torch.nn.init.xavier_normal_(self.cluster_layer.data)#Xavier均匀分布
#
#         # degree
#         self.v = v
#     def forward(self, x):
#         # DNN Module
#         x_bar, z = self.ae(x)
#         q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
#         q = q.pow((self.v + 1.0) / 2.0)
#         q = (q.t() / torch.sum(q, 1)).t()
#         return x_bar, z, q #Q软分布，x_bar重构，z DNN前向最后层输出

# acc_reuslt = []
# nmi_result = []
# ari_result = []
# f1_result = []
#
# def train_idec(dataset):
#     model = IDEC(500, 500, 2000, 2000, 500, 500,
#                  n_input=args.n_input,#155行预定义，args作为宏观变量
#                  n_z=args.n_z,
#                  n_clusters=args.n_clusters,
#                  v=1.0).to(device)
#     print(model)
#
#     optimizer = Adam(model.parameters(), lr=args.lr)
#
#
#     data = torch.Tensor(dataset.x).to(device)
#     y = dataset.y
#     with torch.no_grad(): #此部分不需要在方向时计算梯度
#         x_bar, z, q = model(data)
#
#     kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
#     y_pred = kmeans.fit_predict(z.data.cpu().numpy())
#     y_pred_last = y_pred
#     model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
#     eva(y, y_pred, 'Initialization')
#
#     for epoch in range(200):
#         if epoch % 1 == 0:
#             # update_interval，更新实验结果、目标分布P
#             x_bar, z, tmp_q = model(data)
#             tmp_q = tmp_q.data
#             p = target_distribution(tmp_q)  # 计算p是一种自增强
#             res1 = tmp_q.cpu().numpy().argmax(1)  # Q
#
#             acc, nmi, ari, f1 = eva(y, res1, epoch)
#             acc_reuslt.append(acc)
#             nmi_result.append(nmi)
#             ari_result.append(ari)
#             f1_result.append(f1)
#
#         x_bar, z, q = model(data)
#
#         kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
#         re_loss = F.mse_loss(x_bar, data)
#
#         loss = re_loss + 0.1 * kl_loss
#         print('{} loss: {}'.format(epoch, loss))
#         #loss = kl_loss
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     print("ACC: {:.4f}".format(max(acc_reuslt)))  # np.where(condition)输出满足条件元素的坐标(tuple形式)
#     print("NMI: {:.4f}".format(nmi_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]))
#     print("ARI: {:.4f}".format(ari_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]))
#     print("F1: {:.4f}".format(f1_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]))
#     print("Epoch:", np.where(acc_reuslt == np.max(acc_reuslt))[0][0])
#
#
# #from warnings import simplefilter
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description='train',
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('--name', type=str, default='acm')
#     parser.add_argument('--k', type=int, default=3)
#     parser.add_argument('--lr', type=float, default=1e-3)
#     parser.add_argument('--n_clusters', default=3, type=int)
#     parser.add_argument('--n_z', default=10, type=int)
#     parser.add_argument('--pretrain_path', type=str, default='pkl')
#     args = parser.parse_args()
#     #args.cuda = torch.cuda.is_available()
#     #args.cpu = torch.cpu.is_available()
#     #print("use cuda: {}".format(args.cuda))
#     device = torch.device("cpu")
#
#     #device = torch.device("cuda" if args.cuda else "cpu")
#
#     #simplefilter(action='ignore', category=FutureWarning)
#
#     args.pretrain_path = 'data/{}.pkl'.format(args.name)
#     dataset = load_data(args.name)
#
#     if args.name == 'usps':
#         args.n_clusters = 10
#         args.n_input = 256
#
#     if args.name == 'hhar':
#         args.k = 5
#         args.n_clusters = 6
#         args.n_input = 561
#
#     if args.name == 'reut':
#         args.lr = 1e-4
#         args.n_clusters = 4
#         args.n_input = 2000
#
#     if args.name == 'acm':
#         args.k = None
#         args.n_clusters = 3
#         args.n_input = 1870
#
#     if args.name == 'dblp':
#         args.lr = 1e-3
#         args.k = None
#         args.n_clusters = 4
#         args.n_input = 334
#
#     if args.name == 'cite':
#         args.lr = 1e-3
#         args.k = None
#         args.n_clusters = 6
#         args.n_input = 3703
#
#     print(args)
#     train_idec(dataset)
