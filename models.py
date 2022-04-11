import torch
import torch.nn as nn
from torch.autograd import Variable
import time
from layers import SpGraphAttentionLayer, ConvKB
from torch.nn import functional as F, Parameter
import numpy
from collections import Counter
CUDA = torch.cuda.is_available()  # checking cuda availability
'''
class SpGAT(nn.Module):
    def __init__(self, num_nodes, nfeat, nhid, relation_dim, dropout, alpha, nheads):
        """
            Sparse version of GAT
            nfeat -> Entity Input Embedding dimensions
            nhid  -> Entity Output Embedding dimensions
            relation_dim -> Relation Embedding dimensions
            num_nodes -> number of nodes in the Graph
            nheads -> Used for Multihead attention

        """
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.attentions = [SpGraphAttentionLayer(num_nodes, nfeat,
                                                 nhid,
                                                 relation_dim,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True)
                           for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # W matrix to convert h_input to h_output dimension
        self.W = nn.Parameter(torch.zeros(size=(relation_dim, nheads * nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.out_att = SpGraphAttentionLayer(num_nodes, nhid * nheads,
                                             nheads * nhid, nheads * nhid,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False
                                             )

    def forward(self, Corpus_, batch_inputs, entity_embeddings, relation_embed,
                edge_list, edge_type, edge_embed):
        x = entity_embeddings

        #edge_embed_nhop = relation_embed[
        #    edge_type_nhop[:, 0]] + relation_embed[edge_type_nhop[:, 1]]

        x = torch.cat([att(x, edge_list, edge_embed)
                       for att in self.attentions], dim=1)
        x = self.dropout_layer(x)

        out_relation_1 = relation_embed.mm(self.W)

        edge_embed = out_relation_1[edge_type]
        #edge_embed_nhop = out_relation_1[
        #    edge_type_nhop[:, 0]] + out_relation_1[edge_type_nhop[:, 1]]

        x = F.elu(self.out_att(x, edge_list, edge_embed))
        return x, out_relation_1
        
'''


class HypER(torch.nn.Module):
    def __init__(self, entity_num, input_d, d1, d2):
        super(HypER, self).__init__()
        self.in_channels = 1
        self.out_channels = 32
        self.filt_h = 1
        self.filt_w = 9

        self.inp_drop = torch.nn.Dropout(0.2)
        self.hidden_drop = torch.nn.Dropout(0.3)
        self.feature_map_drop = torch.nn.Dropout2d(0.2)

        self.W_E = nn.Parameter(torch.zeros(size=(input_d, d1)))
        self.bn0 = torch.nn.BatchNorm2d(self.in_channels)
        self.bn1 = torch.nn.BatchNorm2d(self.out_channels)
        self.bn2 = torch.nn.BatchNorm1d(d1)
        self.register_parameter('b', Parameter(torch.zeros(entity_num)))
        fc_length = (1 - self.filt_h + 1) * (d1 - self.filt_w + 1) * self.out_channels
        self.fc = torch.nn.Linear(fc_length, d1)
        fc1_length = self.in_channels * self.out_channels * self.filt_h * self.filt_w
        self.fc1 = torch.nn.Linear(d2, fc1_length)

    def forward(self, head_tail_id, unique_entity_embed):
        unique_entity_embed = unique_entity_embed.mm(self.W_E)
        entity_d = unique_entity_embed.size(1)
        head_embed = unique_entity_embed[head_tail_id[0]].view(-1, 1, 1, entity_d)
        tail_embed = unique_entity_embed[head_tail_id[1]].view(-1, 1, 1, entity_d)
        # [86835,200]

        x = self.bn0(head_embed)
        x = self.inp_drop(x)

        k = self.fc1(tail_embed)
        # [86835,288]
        k = k.view(-1, self.in_channels, self.out_channels, self.filt_h, self.filt_w)
        # [86835,1,32,1,9]
        k = k.view(head_embed.size(0) * self.in_channels * self.out_channels, 1, self.filt_h, self.filt_w)
        # [86835*32,1,1,9]

        x = x.permute(1, 0, 2, 3)
        # [1,86835,1,200]

        x = F.conv2d(x, k, groups=head_embed.size(0))
        # [1,86835*32,1,192]
        x = x.view(head_embed.size(0), 1, self.out_channels, 1 - self.filt_h + 1, head_embed.size(3) - self.filt_w + 1)
        x = x.permute(0, 3, 4, 1, 2)
        # [86835,1,192,1,32]
        x = torch.sum(x, dim=3)
        x = x.permute(0, 3, 1, 2).contiguous()
        # [86835,32,1,192]

        x = self.bn1(x)
        x = self.feature_map_drop(x)
        x = x.view(head_embed.size(0), -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        return x

class SpGAT(nn.Module):
    def __init__(self, num_nodes, nfeat, nhid, relation_dim, dropout, alpha, nheads):
        """
            Sparse version of GAT
            nfeat -> Entity Input Embedding dimensions
            nhid  -> Entity Output Embedding dimensions
            relation_dim -> Relation Embedding dimensions
            num_nodes -> number of nodes in the Graph
            nheads -> Used for Multihead attention

        """
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.attentions = [SpGraphAttentionLayer(num_nodes, nfeat,
                                                 nhid,
                                                 relation_dim,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True)
                           for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # W matrix to convert h_input to h_output dimension(hsm)改W为W_1
        self.W_1 = nn.Parameter(torch.zeros(size=(relation_dim, nheads * nhid)))
        nn.init.xavier_uniform_(self.W_1.data, gain=1.414)

        # hsm 添加对实体的线性转换矩阵
        self.W_2 = nn.Parameter(torch.zeros(size=(relation_dim, nheads * nhid)))

        nn.init.xavier_uniform_(self.W_2.data, gain=1.414)

        self.out_att = SpGraphAttentionLayer(num_nodes, nhid * nheads,
                                         nheads * nhid, nheads * nhid,
                                         dropout=dropout,
                                         alpha=alpha,
                                         concat=False
                                         )


    def forward(self, Corpus_, batch_inputs, unique_entity_embed, unique_relation_embed ,
                edge_list, edge_type, edge_embed):
        #对h进行更新
        x = torch.cat([att(unique_entity_embed, edge_list, edge_embed, edge_type, flag=False)
                       for att in self.attentions], dim=1)
        x = self.dropout_layer(x)
        out_unique_relation_embed = unique_relation_embed.mm(self.W_1)
        out_relation_embed_1 = out_unique_relation_embed[edge_type]
        x = F.elu(self.out_att(x, edge_list, out_relation_embed_1, edge_type, flag = False))



        #(hsm)对r进行更新

        # x_r = torch.cat([att(unique_entity_embed, edge_list, edge_embed, edge_type, flag=True)  # flag=True代表对关系做变换
        #                 for att in self.attentions], dim=1)
        # x_r = self.dropout_layer(x_r[0:unique_relation_embed.shape[0]])
        #
        # out_entity_embed = unique_entity_embed.mm(self.W_2)  # 对实体做变换，和更新头实体时不一样的参数
        # out_relation_embed_2 = x_r[edge_type]
        # x_r = F.elu(self.out_att(out_entity_embed, edge_list, out_relation_embed_2, edge_type, flag = True))
        # x_r = x_r[0:unique_relation_embed.shape[0]]#记得修改长度为len()
        return x

class SpKBGATModified(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, alpha, nheads_GAT):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.nheads_GAT_1 = nheads_GAT[0]
        self.entity_out_dim_2 = entity_out_dim[1]
        self.nheads_GAT_2 = nheads_GAT[1]

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim_1 = relation_out_dim[0]

        self.drop_GAT = drop_GAT
        self.alpha = alpha      # For leaky relu

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.entity_embeddings = nn.Parameter(initial_entity_emb)
        self.relation_embeddings = nn.Parameter(initial_relation_emb)

        self.sparse_gat_1 = SpGAT(self.num_nodes, self.entity_in_dim, self.entity_out_dim_1, self.relation_dim,
                                  self.drop_GAT, self.alpha, self.nheads_GAT_1)

        self.W_entities = nn.Parameter(torch.zeros(
            size=(self.entity_in_dim, self.entity_out_dim_1 * self.nheads_GAT_1)))
        nn.init.xavier_uniform_(self.W_entities.data, gain=1.414)

        #(hsm hyper)
        self.hyper = HypER(self.num_nodes, self.entity_in_dim, self.final_entity_embeddings.size(1), self.final_relation_embeddings.size(1))

    def forward(self, Corpus_, adj, batch_inputs):
        # getting edge list
        edge_list = adj[0]
        edge_type = adj[1]

        if(CUDA):
            edge_list = edge_list.cuda()
            edge_type = edge_type.cuda()

        edge_embed = self.relation_embeddings[edge_type]

        start = time.time()

        self.entity_embeddings.data = F.normalize(
            self.entity_embeddings.data, p=2, dim=1).detach()#什么时候需要normalize
        # self.relation_embeddings.data = F.normalize(
        #     self.relation_embeddings.data, p=2, dim=1).detach()#什么时候需要normalize

        out_entity_1 = self.sparse_gat_1(
            Corpus_, batch_inputs, self.entity_embeddings, self.relation_embeddings,
            edge_list, edge_type, edge_embed)

        #hyper更新关系
        hyper_out_relation =self.hyper(edge_list, self.entity_embeddings)
        coordinate = torch.cat((edge_type.unsqueeze(0), edge_list[0].unsqueeze(0)), dim=0)
        out_relation_1 = torch.sparse_coo_tensor(coordinate, hyper_out_relation, torch.Size(
            [self.relation_embeddings.size(0), hyper_out_relation.size(0),
             hyper_out_relation.size(1)]))  # out_feature=200 E:unique实体的个数
        hyper_out_relation = torch.sparse.sum(out_relation_1, dim=1).to_dense()
        relation = numpy.array(edge_type.cpu())
        relation_classification = list(Counter(relation).values())
        relation_classification = torch.tensor(relation_classification).unsqueeze(-1).cuda()
        out_relation = hyper_out_relation.div(relation_classification)

        #对实体做self-loop
        mask_indices = torch.unique(batch_inputs[:, 2]).cuda()
        mask = torch.zeros(self.entity_embeddings.shape[0]).cuda()
        mask[mask_indices] = 1.0

        entities_upgraded = self.entity_embeddings.mm(self.W_entities)
        out_entity_1 = entities_upgraded + \
            mask.unsqueeze(-1).expand_as(out_entity_1) * out_entity_1

        #out_relation要不要做normalize
        out_entity_1 = F.normalize(out_entity_1, p=2, dim=1)
        out_relation = F.normalize(out_relation, p=2, dim=1)

        self.final_entity_embeddings.data = out_entity_1.data
        self.final_relation_embeddings.data = out_relation.data

        return out_entity_1, out_relation


class SpKBGATConvOnly(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, drop_conv, alpha, alpha_conv, nheads_GAT, conv_out_channels):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.nheads_GAT_1 = nheads_GAT[0]
        self.entity_out_dim_2 = entity_out_dim[1]
        self.nheads_GAT_2 = nheads_GAT[1]

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim_1 = relation_out_dim[0]

        self.drop_GAT = drop_GAT
        self.drop_conv = drop_conv
        self.alpha = alpha      # For leaky relu
        self.alpha_conv = alpha_conv
        self.conv_out_channels = conv_out_channels

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.convKB = ConvKB(self.entity_out_dim_1 * self.nheads_GAT_1, 3, 1,
                             self.conv_out_channels, self.drop_conv, self.alpha_conv)

    def forward(self, Corpus_, adj, batch_inputs):
        conv_input = torch.cat((self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.final_relation_embeddings[
            batch_inputs[:, 1]].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)
        out_conv = self.convKB(conv_input)
        return out_conv

    def batch_test(self, batch_inputs):
        conv_input = torch.cat((self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.final_relation_embeddings[
            batch_inputs[:, 1]].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)
        out_conv = self.convKB(conv_input)
        return out_conv

