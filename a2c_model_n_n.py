import torch
import torch.nn as nn
import numpy as np
# from torch_geometric.graphgym import GCNConv
from torch_geometric.nn import GCNConv, SAGEConv, graclus, avg_pool, global_mean_pool


# Deep neural network that models the DRL agent
class A2CModel(torch.nn.Module):
    def __init__(self, units):
        super(A2CModel, self).__init__()

        self.units = units
        self.common_layers = 1
        self.critic_layers = 1
        self.actor_layers = 1
        self.activation = torch.tanh

        self.conv_first = SAGEConv(self.units, self.units)  # in_feats, out_feats,代表源、目标节点特征维度
        self.conv_common = nn.ModuleList(  # SAGEConv是GraphSAGE的操作器，输入输出是顶点的向量维度，比如Bert512
            [SAGEConv(self.units, self.units)
             for i in range(self.common_layers)]
        )
        # self.conv_actor = nn.ModuleList(
        #     [SAGEConv(self.units,
        #               1 if i == self.actor_layers - 1 else self.units)
        #      for i in range(self.actor_layers)]
        # )
        self.conv_actor = nn.ModuleList(
            [SAGEConv(self.units, self.units)
             for i in range(self.actor_layers)]
        )
        self.final_actor = nn.Linear(self.units * 2, 1)
        self.conv_critic = nn.ModuleList(
            [SAGEConv(self.units, self.units)
             for i in range(self.critic_layers)]
        )
        self.final_critic = nn.Linear(self.units, 1)

    def forward(self, x, edge_index, batch):
        # do_not_flip = torch.where(x[:, 2] != 0.)  # mask
        x = self.activation(self.conv_first(x, edge_index))  # tanh
        for i in range(self.common_layers):
            x = self.activation(self.conv_common[i](x, edge_index))
        # SAGEConv的forward方法(DGLGraph,feat : torch.Tensor or pair of torch.Tensor)
        x_actor = x  # actor branch
        for i in range(self.actor_layers):
            x_actor = self.conv_actor[i](x_actor, edge_index)
            if i < self.actor_layers - 1:
                x_actor = self.activation(x_actor)
        # x_actor[do_not_flip] = torch.tensor(-np.Inf)
        sen_len = x.shape[0]
        edge_actor = []
        for i in range(sen_len):
            temp = x_actor[i].repeat(sen_len, 1)
            x_temp = torch.cat((temp, x_actor), dim=-1)
            if edge_actor == []:
                edge_actor = x_temp
            else:
                edge_actor = torch.cat((edge_actor, x_temp), dim=0)
        edge_actor = self.final_actor(edge_actor)
        # x_actor = torch.log_softmax(x_actor, dim=0)  # actor在最后都过了一个softmax，转换成概率tensor
        edge_actor = torch.log_softmax(edge_actor, dim=0)

        if not self.training:
            return edge_actor

        x_critic = x.detach()  # 复制操作，detach后的数据和原tensor共享内存
        for i in range(self.critic_layers):
            x_critic = self.conv_critic[i](x_critic, edge_index)
            if i < self.critic_layers - 1:
                x_critic = self.activation(x_critic)
        # 边表示为首尾顶点和的平均值
        edge_critic = []
        for i in range(sen_len):
            temp_critic = x_critic[i].repeat(sen_len, 1)
            temp_critic = (temp_critic + x_critic) / 2
            if edge_critic == []:
                edge_critic = temp_critic
            else:
                edge_critic = torch.cat((edge_critic, temp_critic), dim=0)
        edge_critic = self.final_critic(edge_critic)
        batch = torch.zeros(sen_len*sen_len).cuda()
        batch = batch.to(dtype=torch.int64)
        edge_critic = torch.tanh(global_mean_pool(edge_critic, batch))
        # edge_critic = torch.tanh(edge_critic)  # critic 变成一个(-1,1)区间的标量
        temp_critic = None
        x_temp = None
        x = None
        edge_index = None
        return edge_actor, edge_critic
