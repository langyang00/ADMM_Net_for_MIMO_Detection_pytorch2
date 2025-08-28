import torch
import numpy as np

def loss(x_true, x_hat, x_result, K, mod):
    if mod == 0:
        loss_val = torch.mean(torch.sum(torch.square(x_true - x_hat), dim=1))
        # 对于 BPSK，x_true 和 x_result 都是实数
        ser = torch.mean(torch.mean((x_true != x_result).float(), dim=1))
    else:
        # PyTorch 没有 tf.complex 的直接对应物，需要手动处理复数
        K_half = K // 2
        X_c = x_true[:, :K_half] + 1j * x_true[:, K_half:K]
        x_c = x_result[:, :K_half] + 1j * x_result[:, K_half:K]

        loss_val = torch.mean(torch.sum(torch.square(x_true - x_hat), dim=1))
        # 比较复数张量
        ser = torch.mean(torch.mean((X_c != x_c).float(), dim=1))

    return loss_val, ser

# 注意：为了简化，这里直接使用 Python 的复数类型。在实际 PyTorch 中，
# 可能会使用 torch.view_as_real 和 torch.view_as_complex 来操作。

def demod_nonlinear(x, theta, mod):
    if mod == 0 or mod == 4:
        x_hat = torch.tanh(theta * x)
    elif mod == 16:
        x_hat = torch.tanh(theta * x) + torch.tanh(theta * (x - 2)) + torch.tanh(theta * (x + 2))
    elif mod == 64:
        x_hat = torch.tanh(x) + \
                torch.tanh(theta * (x - 2)) + torch.tanh(theta * (x + 2)) + \
                torch.tanh(theta * (x - 4)) + torch.tanh(theta * (x + 4)) + \
                torch.tanh(theta * (x - 6)) + torch.tanh(theta * (x + 6))
    return x_hat

def demod_projection(x, mod):
    if mod == 0:
        x_result = torch.sign(x)
    else:
        # torch.clip 代替 tf.clip_by_value
        log2_mod = np.log2(mod)
        x_result = torch.clamp(x, -(log2_mod - 1) - 0.5, (log2_mod - 1) + 0.5)
        x_result = (torch.round((x_result + (log2_mod - 1)) / 2)) * 2 - (log2_mod - 1)
    return x_result
















