import torch
import torch.nn as nn
import numpy as np
import os

from Parameters import L, K_orig, N_orig, mod, train_iter, save_file_name
from function import demod_nonlinear, demod_projection, loss


class Gated_ADMM_Net(nn.Module):
    def __init__(self):
        super(Gated_ADMM_Net, self).__init__()

        # 这里的 K 和 N 必须在初始化时确定
        if mod == 0:
            self.K = K_orig
            self.N = N_orig
        else:
            self.K = 2 * K_orig
            self.N = 2 * N_orig

        # 定义所有可学习参数
        if train_iter == 0:
            # 加载已训练的参数
            param_path_d_t_b = os.path.join(save_file_name,
                                            f'Gated_ADMM_Net_parameters_d_t_b_({K_orig}x{N_orig} {mod}QAM MIMO).npy')
            param_path_W1_W2 = os.path.join(save_file_name,
                                            f'Gated_ADMM_Net_parameters_W1_W2_({K_orig}x{N_orig} {mod}QAM MIMO).npy')
            param_path_b1_b2 = os.path.join(save_file_name,
                                            f'Gated_ADMM_Net_parameters_b1_b2_({K_orig}x{N_orig} {mod}QAM MIMO).npy')

            Gated_ADMM_Net_parameters_d_t_b = np.load(param_path_d_t_b, allow_pickle=True)
            Gated_ADMM_Net_parameters_W1_W2 = np.load(param_path_W1_W2, allow_pickle=True)
            Gated_ADMM_Net_parameters_b1_b2 = np.load(param_path_b1_b2, allow_pickle=True)

            self.delta = nn.Parameter(torch.from_numpy(Gated_ADMM_Net_parameters_d_t_b[0]).float())
            self.theta = nn.Parameter(torch.from_numpy(Gated_ADMM_Net_parameters_d_t_b[1]).float())
            self.beta = nn.Parameter(torch.from_numpy(Gated_ADMM_Net_parameters_d_t_b[2]).float())
            self.W1 = nn.Parameter(torch.from_numpy(Gated_ADMM_Net_parameters_W1_W2[0]).float())
            self.W2 = nn.Parameter(torch.from_numpy(Gated_ADMM_Net_parameters_W1_W2[1]).float())
            self.b1 = nn.Parameter(torch.from_numpy(Gated_ADMM_Net_parameters_b1_b2[0]).float())
            self.b2 = nn.Parameter(torch.from_numpy(Gated_ADMM_Net_parameters_b1_b2[1]).float())

        else:
            # 随机初始化可学习参数，保持与原始 TensorFlow 代码一致的维度
            self.delta = nn.Parameter(torch.ones(L, 1) * 0.0001)
            self.theta = nn.Parameter(torch.ones(L, 1) * 1.5)
            self.beta = nn.Parameter(torch.ones(L, 1) * 0.1)
            self.W1 = nn.Parameter(torch.ones(L, 1, self.K + self.N) * 0.001)
            self.W2 = nn.Parameter(torch.ones(L, 1, self.K + self.N) * 0.001)
            self.b1 = nn.Parameter(torch.ones(L, 1, 1) * 0.001)
            self.b2 = nn.Parameter(torch.ones(L, 1, 1) * 0.001)

    def forward(self, H, Y, H_T, X):
        batch_size = H.shape[0]
        N, K = H.shape[1], H.shape[2]

        x = torch.zeros(batch_size, K, 1, device=Y.device)
        e = torch.zeros(batch_size, N, 1, device=Y.device)
        v_hat = torch.zeros(batch_size, N, 1, device=Y.device)
        v_pre = torch.zeros(batch_size, N, 1, device=Y.device)

        c = torch.ones(batch_size, 1, 1, device=Y.device)

        layer_loss = []
        layer_ser = []

        # 核心逻辑
        for i in range(L):
            if i == 0:
                H_T_H = torch.matmul(H_T, H)
                pinv_H = torch.matmul(torch.linalg.pinv(H_T_H), H_T)
                x = torch.matmul(pinv_H, Y)

            x = x - (self.delta[i] * torch.matmul(H_T, e + (1 - 2 * self.beta[i]) * v_hat))
            x = demod_nonlinear(x, self.theta[i], mod)

            e = torch.matmul(H, x) - Y

            # 在这里，我们将 x 和 v_hat 拼接
            state = torch.cat([x, v_hat], dim=1)  # 形状: [batch_size, K+N, 1]

            # 门控网络的计算，保持和原始代码相同的矩阵乘法
            alpha = torch.matmul(self.W1[i], state) + self.b1[i]
            gamma = torch.matmul(self.W2[i], state) + self.b2[i]

            alpha = 2 * torch.sigmoid(alpha)
            gamma = torch.sigmoid(gamma)

            v = alpha * e + (1 - alpha * self.beta[i]) * v_hat
            v_hat = v + gamma * (v - v_pre)

            v_pre = v

            x_hat = demod_nonlinear(x, 100, mod)
            x_result = demod_projection(x_hat, mod)

            LOSS, SER = loss(X, x_hat, x_result, K, mod)
            layer_loss.append(LOSS)
            layer_ser.append(SER)

        return x_hat, x_result, layer_loss, layer_ser, self.delta, self.theta, self.beta, self.W1, self.W2, self.b1, self.b2