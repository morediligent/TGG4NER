import argparse
from pathlib import Path

import networkx as nx

import torch
import torch.nn as nn

from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import SAGEConv, graclus, avg_pool, global_mean_pool
from torch_geometric.utils import to_networkx, k_hop_subgraph, degree

import numpy as np
from numpy import random

import scipy
from scipy.sparse import coo_matrix, rand
from scipy.io import mmread
from scipy.spatial import Delaunay

import copy
import timeit
import os
from itertools import combinations


# Change the feature of the selected vertex
def change_vertex(state, vertex):
    if (state.x[vertex, :2] == torch.tensor([1., 0.])).all():
        state.x[vertex, 0] = torch.tensor(0.)
        state.x[vertex, 1] = torch.tensor(1.)
    else:
        state.x[vertex, 0] = torch.tensor(1.)
        state.x[vertex, 1] = torch.tensor(0.)

    return state


# Reward to train the DRL agent
def reward_NC(state, vertex):
    new_state = state.clone()
    new_state = change_vertex(new_state, vertex)
    return normalized_cut(state) - normalized_cut(new_state)


# Normalized cut of the input graph
def normalized_cut(graph):
    cut, da, db = volumes(graph)
    if da == 0 or db == 0:
        return 2
    else:
        return cut * (1 / da + 1 / db)


# Coarsen a pytorch geometric graph, then find the cut with METIS and
# interpolate it back


def partition_metis_refine(graph):
    cluster = graclus(graph.edge_index)
    coarse_graph = avg_pool(
        cluster,
        Batch(
            batch=graph.batch,
            x=graph.x,
            edge_index=graph.edge_index))
    coarse_graph_nx = to_networkx(coarse_graph, to_undirected=True)
    _, parts = nxmetis.partition(coarse_graph_nx, 2)
    mparts = np.array(parts)
    coarse_graph.x[np.array(parts[0])] = torch.tensor([1., 0.])
    coarse_graph.x[np.array(parts[1])] = torch.tensor([0., 1.])
    _, inverse = torch.unique(cluster, sorted=True, return_inverse=True)
    graph.x = coarse_graph.x[inverse]
    return graph


# Subgraph around the cut


# Training loop
def training_loop(
        model,
        training_dataset,
        episodes,
        gamma,
        time_to_sample,
        coeff,
        optimizer,
        print_loss,
        k):
    # Here start the main loop for training
    for i in range(episodes):
        rew_partial = 0
        p = 0
        print('epoch:', i)
        for graph in training_dataset:
            print('Graph:', p, '  Number of nodes:', graph.num_nodes)
            start_all = partition_metis_refine(graph)

            data = k_hop_graph_cut(start_all, k)
            graph_cut, positions = data[0], data[1]
            len_episode = cut(graph)

            start = graph_cut
            time = 0

            rews, vals, logprobs = [], [], []
            # Here starts the episod related to the graph "start"
            while time < len_episode:
                # we evaluate the A2C agent on the graph
                policy, values = model(start)
                probs = policy.view(-1)  # 转换成一维张量
                # 按照传入的probs中给定的概率，在相应的位置处进行取样，取样返回的是该位置的整数索引
                action = torch.distributions.Categorical(
                    logits=probs).sample().detach().item()  # 取样

                # compute the reward associated with this action
                rew = reward_NC(start_all, positions[action])
                rew_partial += rew
                # Collect the log-probability of the chosen action
                logprobs.append(policy.view(-1)[action])
                # Collect the value of the chosen action
                vals.append(values)
                # Collect the reward
                rews.append(rew)

                new_state = start.clone()
                new_state_orig = start_all.clone()
                # we flip the vertex returned by the policy
                new_state = change_vertex(new_state, action)
                new_state_orig = change_vertex(
                    new_state_orig, positions[action])
                # Update the state
                start = new_state
                start_all = new_state_orig

                _, va, vb = volumes(start_all)

                nnz = start_all.num_edges
                start.x[:, 3] = torch.true_divide(va, nnz)
                start.x[:, 4] = torch.true_divide(vb, nnz)

                time += 1

                # After time_to_sample episods we update the loss
                if i % time_to_sample == 0 or time == len_episode:

                    logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
                    vals = torch.stack(vals).flip(dims=(0,)).view(-1)
                    rews = torch.tensor(rews).flip(dims=(0,)).view(-1)

                    # Compute the advantage
                    R = []
                    R_partial = torch.tensor([0.])
                    for j in range(rews.shape[0]):
                        R_partial = rews[j] + gamma * R_partial
                        R.append(R_partial)

                    R = torch.stack(R).view(-1)
                    advantage = R - vals.detach()

                    # Actor loss
                    actor_loss = (-1 * logprobs * advantage)

                    # Critic loss
                    critic_loss = torch.pow(R - vals, 2)

                    # Finally we update the loss
                    optimizer.zero_grad()

                    loss = torch.mean(actor_loss) + \
                           torch.tensor(coeff) * torch.mean(critic_loss)

                    rews, vals, logprobs = [], [], []

                    loss.backward()

                    optimizer.step()
            if p % print_loss == 0:
                print('graph:', p, 'reward:', rew_partial)
            rew_partial = 0
            p += 1

    return model


# Deep neural network that models the DRL agent
class A2CModel(torch.nn.Module):
    def __init__(self, units):
        super(A2CModel, self).__init__()

        self.units = units
        self.common_layers = 1
        self.critic_layers = 1
        self.actor_layers = 1
        self.activation = torch.tanh

        self.conv_first = SAGEConv(self.units, self.units)
        self.conv_common = nn.ModuleList(  # SAGEConv是GraphSAGE的操作器，输入输出是顶点的向量维度，比如Bert512
            [SAGEConv(self.units, self.units)
             for i in range(self.common_layers)]
        )
        self.conv_actor = nn.ModuleList(
            [SAGEConv(self.units,
                      1 if i == self.actor_layers - 1 else self.units)
             for i in range(self.actor_layers)]
        )
        self.conv_critic = nn.ModuleList(
            [SAGEConv(self.units, self.units)
             for i in range(self.critic_layers)]
        )
        self.final_critic = nn.Linear(self.units, 1)

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch

        do_not_flip = torch.where(x[:, 2] != 0.)  # mask
        x = self.activation(self.conv_first(x, edge_index))  # tanh
        for i in range(self.common_layers):
            x = self.activation(self.conv_common[i](x, edge_index))

        x_actor = x  # actor branch
        for i in range(self.actor_layers):
            x_actor = self.conv_actor[i](x_actor, edge_index)
            if i < self.actor_layers - 1:
                x_actor = self.activation(x_actor)
        x_actor[do_not_flip] = torch.tensor(-np.Inf)
        x_actor = torch.log_softmax(x_actor, dim=0)  # actor在最后都过了一个softmax，转换成概率tensor

        if not self.training:
            return x_actor

        x_critic = x.detach()  # 复制操作，detach后的数据和原tensor共享内存
        for i in range(self.critic_layers):
            x_critic = self.conv_critic[i](x_critic, edge_index)
            if i < self.critic_layers - 1:
                x_critic = self.activation(x_critic)
        x_critic = self.final_critic(x_critic)
        x_critic = torch.tanh(global_mean_pool(x_critic, batch))  # critic 变成一个(-1,1)标量？
        return x_actor, x_critic  # 仍然不清楚 x_actor, x_critic怎么定义

    '''
	def forward_c(self, graph, gcsr):
		n = gcsr.shape[0]
		x_actor = torch.zeros([n, 1], dtype=torch.float32)
		libcdrl.forward(
		    ctypes.c_int(n),
		    ctypes.c_void_p(gcsr.indptr.ctypes.data),
		    ctypes.c_void_p(gcsr.indices.ctypes.data),
		    ctypes.c_void_p(graph.x.data_ptr()),
		    ctypes.c_void_p(x_actor.data_ptr()),
		    ctypes.c_void_p(self.conv_first.lin_l.weight.data_ptr()),
		    ctypes.c_void_p(self.conv_first.lin_r.weight.data_ptr()),
		    ctypes.c_void_p(self.conv_common[0].lin_l.weight.data_ptr()),
		    ctypes.c_void_p(self.conv_common[0].lin_r.weight.data_ptr()),
		    ctypes.c_void_p(self.conv_actor[0].lin_l.weight.data_ptr()),
		    ctypes.c_void_p(self.conv_actor[0].lin_r.weight.data_ptr())
		)
		
		return x_actor
	'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--out', default='./temp_edge/', type=str)
    parser.add_argument(
        "--lr",
        default=0.001,
        help="Learning rate",
        type=float)
    parser.add_argument(
        "--units",
        default=5,
        help="Number of units in conv layers",
        type=int)
	parser.add_argument(
		"--dataset",
		default='weibo',
		type=str)
    args = parser.parse_args()

    torch.manual_seed(1)
    np.random.seed(2)

    model = A2CModel(args.units)

    print(model)
    print('Model parameters:', sum([w.nelement() for w in model.parameters()]))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print('Start training')

    t0 = timeit.default_timer()
    training_loop(model, args.dataset, args.epochs, gamma, time_to_sample, coeff, optimizer, print_loss, hops)
    ttrain = timeit.default_timer() - t0

    print('Training took:', ttrain, 'seconds')

    # Saving the model
    torch.save(model.state_dict(), args.out + '/' + 'model_a2c_mask')