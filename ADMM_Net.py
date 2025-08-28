import torch
import torch.nn as nn
import numpy as np

# 假设已经定义了 Parameters
from Parameters import *
from function import *

class ADMM_Net(nn.Module):
    def __init__(self):
        super(ADMM_Net, self).__init__()
        # 在 __init__ 中定义可学习参数
        if train_iter == 0:
            parameters = np.load(save_file_name + f'/ADMM_Net_parameters_({K_orig}x{N_orig} {mod}QAM MIMO).npy',
                                 allow_pickle=True)
            self.delta = nn.Parameter(torch.from_numpy(parameters[0]).float())
            self.theta = nn.Parameter(torch.from_numpy(parameters[1]).float())
            self.alpha = nn.Parameter(torch.from_numpy(parameters[2]).float())
            self.beta = nn.Parameter(torch.from_numpy(parameters[3]).float())
            self.gamma = nn.Parameter(torch.from_numpy(parameters[4]).float())
        else:
            self.delta = nn.Parameter(torch.ones(L, 1) * 0.001)
            self.theta = nn.Parameter(torch.ones(L, 1) * 1.5)
            self.alpha = nn.Parameter(torch.ones(L, 1) * 1.5)
            self.beta = nn.Parameter(torch.ones(L, 1) * 0.1)
            self.gamma = nn.Parameter(torch.from_numpy(init_gamma).float().squeeze())

    def forward(self, H, Y, H_T, X):
        # 这里的 K 和 N 是在主程序中计算的
        batch_size = H.shape[0]
        N, K = H.shape[1], H.shape[2]

        x = torch.zeros(batch_size, K, 1, device=Y.device)
        e = torch.zeros(batch_size, N, 1, device=Y.device)
        v_hat = torch.zeros(batch_size, N, 1, device=Y.device)
        v_pre = torch.zeros(batch_size, N, 1, device=Y.device)

        layer_loss = []
        layer_ser = []

        # 迭代 L 层
        for i in range(L):
            if i == 0:
                # 伪逆计算
                H_T_H = torch.matmul(H_T, H)
                pinv_H = torch.matmul(torch.linalg.inv(H_T_H), H_T)
                x = torch.matmul(pinv_H, Y)

            x = x - (self.delta[i] * torch.matmul(H_T, e + (1 - 2 * self.beta[i]) * v_hat))
            x = demod_nonlinear(x, self.theta[i], mod)

            e = torch.matmul(H, x) - Y

            v = self.alpha[i] * e + (1 - self.alpha[i] * self.beta[i]) * v_hat
            v_hat = v + self.gamma[i] * (v - v_pre)

            v_pre = v

            x_hat = demod_nonlinear(x, 100, mod)
            x_result = demod_projection(x_hat, mod)

            LOSS, SER = loss(X, x_hat, x_result, K, mod)
            layer_loss.append(LOSS)
            layer_ser.append(SER)

        return x_hat, x_result, layer_loss, layer_ser, self.delta, self.theta, self.alpha, self.beta, self.gamma