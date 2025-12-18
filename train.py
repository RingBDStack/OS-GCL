import argparse
import gc
import os.path as osp
import random
import sys
from time import perf_counter as t
import yaml
from yaml import SafeLoader

import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, CitationFull, Amazon, Coauthor
from torch_geometric.utils import dropout_adj, to_scipy_sparse_matrix
from torch_geometric.nn import GCNConv

from model import Encoder, GaussianModel, generate_augmented_gaussian_distributions
from eval import label_classification, evaluation
# import inspect
# from gpu_mem_track import MemTracker  # 引用显存跟踪代码
import os
import numpy as np
import scipy.sparse as sp


def train(model: GaussianModel, x, new_edge):
    delta = args.delta
    beta = args.beta
    mean_range = args.mean_range
    std_range = args.std_range
    lamda = args.lamda

    model.train()
    optimizer.zero_grad()
    z_mean, z_std = model(x, new_edge, beta, delta)
    (z1_mean, z1_std), (z2_mean, z2_std) = generate_augmented_gaussian_distributions(z_mean, z_std, mean_range * z_mean,
                                                                                     std_range * z_std)
    loss = model.loss(z1_mean, z2_mean, z1_std, z2_std, lamda).to(device)
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model: GaussianModel, x, new_edge, y, name, data, final=False):
    model.eval()
    delta = args.delta
    beta = args.beta
    z, _ = model(x, new_edge, beta, delta)  # embeddings
    dic = label_classification(z, y, 0.2, name, data)  # 标签分类
    F1Mi = dic['F1Mi']
    F1Ma = dic['F1Ma']
    return F1Mi, F1Ma

def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'computers', 'photo', 'CS', 'Physics']
    name = 'dblp' if name == 'DBLP' else name

    if name == 'CS' or name == 'Physics':
        return Coauthor(
            path,
            name, transform=T.NormalizeFeatures())

    if name == 'computers' or name == 'photo':
        return Amazon(
            path,
            name, transform=T.NormalizeFeatures())

    return (CitationFull if name == 'dblp' else Planetoid)(
        path,
        name, transform=T.NormalizeFeatures())



if __name__ == '__main__':
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global config_wandb

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--tau1', type=float, default=0.5)
    parser.add_argument('--tau2', type=float, default=0.4)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--delta', type=float, default=0.6)
    parser.add_argument('--lamda', type=float, default=0.1)
    parser.add_argument('--num_hidden', type=int, default=256)
    parser.add_argument('--num_proj_hidden', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--mean_range', type=float, default=0.04)
    parser.add_argument('--std_range', type=float, default=0.035)
    args = parser.parse_args()  # 保存

    # assert args.gpu_id in range(0, 8)
    # torch.cuda.set_device(args.gpu_id)
    torch.cuda.set_device(-1)

    learning_rate = args.lr
    num_hidden = args.num_hidden
    num_proj_hidden = args.num_proj_hidden
    num_layers = args.num_layers
    activation = nn.PReLU()
    base_model = GCNConv
    num_epochs = 1000
    weight_decay = args.wd
    tau1 = args.tau1
    tau2 = args.tau2
    k = args.k

    path = osp.join(osp.expanduser('~'), 'datasets', args.dataset)  # 创建数据集的路径
    dataset = get_dataset(path, args.dataset)
    data = dataset[0]  # 图
    data = data.to(device)

    coo1 = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.x.shape[0])
    coo1 = coo1.toarray()
    tmp = coo1.copy()

    for i in range(2, k + 1):
        coo1 += tmp ** i
    coo1 = sp.coo_matrix(coo1)  # 包含边数
    new_edge = coo1.toarray()
    new_edge = torch.tensor(new_edge).to(device)

    del tmp, coo1
    torch.cuda.empty_cache()
    gc.collect()

    # save_path
    cur_path = osp.abspath(__file__)
    cur_dir = osp.dirname(cur_path)
    model_save_path = osp.join(cur_dir, args.dataset + '.pkl')

    model = GaussianModel(dataset.num_features, num_hidden, activation, base_model, num_layers, num_proj_hidden,
                          tau1, tau2).to(device)
    global optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = t()
    prev = start
    if not args.test:
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(1, num_epochs + 1):
            loss = train(model, data.x, new_edge)
            now = t()
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
                  f'this epoch {now - prev:.4f}, total {now - start:.4f}')
            prev = now
            if epoch % 100 == 0:
                F1Mi, F1Ma = test(model, data.x, new_edge, data.y, args.dataset, data, final=True)

    if args.test:
        if osp.exists(model_save_path):
            model.load_state_dict(torch.load(model_save_path))
            model.eval()
            F1Mi, F1Ma = test(model, data.x, data.edge_index, data.y, args.dataset, data, final=True)
            print(F1Mi, F1Ma)
        else:
            print('model not exit')
            sys.exit(0)


