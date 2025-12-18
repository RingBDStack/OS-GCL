import gc
from typing import Tuple, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray, dtype, floating, complexfloating
from torch_geometric.nn import GCNConv
import scipy.sparse as sp
import numpy as np

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.ln = nn.LayerNorm(nb_classes)
        self.weights_init(self.fc)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        ret = self.ln(ret)
        return ret


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2):  # 输入(出）特征的通道数，激活函数，基础模型（默认为 GCNConv），图卷积层的数量（默认为 2）
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]  # 创建一个存储图卷积层的列表，并初始化第一层图卷积层
        for _ in range(1, k - 1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)
        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index, edge_weight))
        return x


class GaussianModel(torch.nn.Module):
    def __init__(self, num_fea: int, num_hidden: int, activation, base_model, num_layers, num_proj_hidden: int,
                 tau1: float, tau2: float):
        super(GaussianModel, self).__init__()
        self.mean_fc = torch.nn.Linear(num_fea, num_fea)
        self.std_fc = torch.nn.Linear(num_fea, num_fea)

        self.mean_encoder = Encoder(num_fea, num_hidden, activation, base_model=base_model, k=num_layers)
        self.std_encoder = Encoder(num_fea, num_hidden, activation, base_model=base_model, k=num_layers)
        self.tau1: float = tau1
        self.tau2: float = tau2

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor, new_edge: torch.Tensor, beta, delta, mean_range:float = 0.0, std_range:float = 0.0) -> torch.Tensor:
        x_mean = self.mean_fc(x)
        x_std = self.std_fc(x)
        term = (1 - beta) * F.normalize(WS_v22(x_mean, x_mean, x_std, x_std)) + new_edge * beta
        term = torch.clamp(term, min=1e-6, max=1 - 1e-6)
        term = torch.log(term / (1 - term))
        eps = torch.rand(new_edge.shape[0], new_edge.shape[0], device=new_edge.device)
        eps = torch.clamp(eps, min=1e-6, max=1 - 1e-6)
        term += torch.log(eps / (1 - eps))
        term = torch.sigmoid(term / self.tau2)
        term = torch.where(term > delta, term, torch.zeros_like(term))

        del eps
        torch.cuda.empty_cache()
        gc.collect()
        edge_weight = torch.nonzero(term, as_tuple=False)
        edge_index = edge_weight.t().contiguous().detach()
        edge_weight = term[edge_weight[:, 0], edge_weight[:, 1]]

        z_mean = self.mean_encoder(x_mean, edge_index, edge_weight)
        z_std = self.std_encoder(x_std, edge_index, edge_weight)

        return z_mean, z_std

    def projection(self, z: torch.Tensor):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau1)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def Gaussian_semi_loss(self, z1_mean, z2_mean, z1_std, z2_std):
        f = lambda x: torch.exp(x / self.tau1)
        ws2 = WS_diag(z1_mean, z2_mean, z1_std, z2_std)
        ws1 = ws_per(z1_mean, z2_mean, z1_std, z2_std)
        ws1 = f(ws1)
        ws2 = f(ws2)
        return -torch.log(
            ws2
            / ((z1_mean.shape[0] - 1) * ws1 + ws2))

    def loss(self, z1_mean: torch.Tensor, z2_mean: torch.Tensor,
             z1_std: torch.Tensor, z2_std: torch.Tensor, lamda):

        z1_mean = self.projection(z1_mean)
        z2_mean = self.projection(z2_mean)
        z1_std = self.projection(z1_std)
        z2_std = self.projection(z2_std)

        l = self.Gaussian_semi_loss(z1_mean, z2_mean, z1_std, z2_std)
        l += self.Gaussian_semi_loss(z2_mean, z1_mean, z2_std, z1_std)

        kld_loss1 = torch.mean(-0.5 * torch.sum(1 + z1_std - z1_mean ** 2 - z1_std.exp(), dim=1), dim=0)
        kld_loss2 = torch.mean(-0.5 * torch.sum(1 + z2_std - z2_mean ** 2 - z2_std.exp(), dim=1), dim=0)

        return (l * 0.5).mean() + lamda * (0.5 * kld_loss1 + 0.5 * kld_loss2)

def generate_augmented_gaussian_distributions(original_mean, original_std, mean_range, std_range):
    noise_mean1 = (torch.rand_like(original_mean) * (mean_range * 2)) - mean_range
    noise_std1 = (torch.rand_like(original_std) * (std_range * 2)) - std_range
    noise_mean2 = (torch.rand_like(original_mean) * (mean_range * 2)) - mean_range
    noise_std2 = (torch.rand_like(original_std) * (std_range * 2)) - std_range

    new_mean1 = original_mean + noise_mean1
    new_mean2 = original_mean + noise_mean2
    new_std1 = original_std + noise_std1
    new_std2 = original_std + noise_std2

    del noise_mean1, noise_std1, noise_mean2, noise_std2
    torch.cuda.empty_cache()
    gc.collect()

    return (new_mean1, new_std1), (new_mean2, new_std2)




def WS_v22(z1_mean, z2_mean, z1_std, z2_std):
    z1_mean = F.normalize(z1_mean)
    z2_mean = F.normalize(z2_mean)
    z1_cov = z1_std.exp()
    z2_cov = z2_std.exp()
    z1_cov = F.normalize(z1_cov)
    z2_cov = F.normalize(z2_cov)

    res = (torch.cdist(z1_mean, z2_mean, p=2) ** 2
           + ((torch.sum(z1_cov, dim=1)).unsqueeze(1)).repeat(1, z1_std.size(0))
           + torch.sum(z2_cov, dim=1).repeat(z2_std.size(0), 1) - (
                   ((z1_cov).sqrt()) @ ((z2_cov).sqrt()).T) * 2)

    res = (-res).exp()

    # res = - res
    # res = (res - torch.min(res)) / (torch.max(res) - torch.min(res))
    # res = torch.tanh((res-0.5) * 2)
    return res


def ws_per(z1_mean, z2_mean, z1_std, z2_std):
    z1_mean = F.normalize(z1_mean)
    z2_mean = F.normalize(z2_mean)
    z1_cov = z1_std.exp()
    z2_cov = z2_std.exp()
    z1_cov = F.normalize(z1_cov)
    z2_cov = F.normalize(z2_cov)

    mean1_sum = torch.sum(z1_mean, dim=0)
    mean1_sum = mean1_sum.repeat(z1_mean.shape[0], 1)
    other_mean1 = (mean1_sum - z1_mean) / (z1_mean.shape[0] - 1)
    mean2_sum = torch.sum(z2_mean, dim=0)
    mean2_sum = mean2_sum.repeat(z2_mean.shape[0], 1)
    other_mean2 = (mean2_sum - z2_mean) / (z2_mean.shape[0] - 1)

    cov1_sum = torch.sum(z1_cov, dim=0)
    cov1_sum = cov1_sum.repeat(z1_mean.shape[0], 1)
    other_cov1 = (cov1_sum - z1_cov) / ((z1_mean.shape[0] - 1))
    cov2_sum = torch.sum(z2_cov, dim=0)
    cov2_sum = cov2_sum.repeat(z2_mean.shape[0], 1)
    other_cov2 = (cov2_sum - z2_cov) / ((z2_mean.shape[0] - 1))

    res = (torch.sum((z1_mean - other_mean1) ** 2 + (z1_mean - other_mean2) ** 2, dim=1, keepdim=True)
           + torch.sum(z1_cov + other_cov1 + z1_cov + other_cov2, dim=1, keepdim=True)
           - 2 * torch.sum((z1_cov * other_cov1).sqrt() + (z1_cov * other_cov2).sqrt(), dim=1, keepdim=True)).T

    res = - res
    res = (res - torch.min(res)) / (torch.max(res) - torch.min(res))
    res = torch.tanh((res-0.5) * 2)
    return res


def WS_diag(z1_mean, z2_mean, z1_std, z2_std):
    z1_mean = F.normalize(z1_mean)
    z2_mean = F.normalize(z2_mean)
    z1_cov = z1_std.exp()
    z2_cov = z2_std.exp()
    z1_cov = F.normalize(z1_cov)
    z2_cov = F.normalize(z2_cov)

    term1 = torch.sum((z1_mean - z2_mean) ** 2, dim=1)
    term2 = torch.sum(z1_cov + z2_cov, dim=1)
    term3 = torch.sum((z1_cov * z2_cov).sqrt(), dim=1)
    res = term1 + term2 - 2 * term3
    res = - res
    res = (res - torch.min(res)) / (torch.max(res) - torch.min(res))
    res = torch.tanh((res-0.5) * 4)
    return res


