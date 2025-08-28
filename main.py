import torch
import torch.optim as optim
import time as tm
import numpy as np
import datetime
import os

from Parameters import *
from Data_generation import generate_data
from function import loss, demod_projection  # 确保这些函数已经转换为 PyTorch
# 导入网络模型
from ADMM_Net import ADMM_Net as ADMM
from Gated_ADMM_Net import Gated_ADMM_Net as Gated_ADMM

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(save_file_name):
    os.makedirs(save_file_name)

if mod == 0:
    K = K_orig
    N = N_orig
else:
    K = 2 * K_orig
    N = 2 * N_orig

# 初始化模型
if detector_name == "ZF":
    # ZF 是一种非训练的算法，直接在循环中实现
    pass
elif detector_name == "ADMM":
    model = ADMM().to(device)
elif detector_name == "Gated_ADMM":
    model = Gated_ADMM().to(device)

print(
    f'Tx:{K_orig}\t Rx:{N_orig}\nModulation:{mod} QAM (BPSK = 0 QAM)\nMIMO detector:{detector_name}\n\nTrain iteration:{train_iter}\nTrain batch size:{train_batch_size}\n')
print(f'Test iteration:{test_iter}\nTest batch size:{test_batch_size}\n')

# 训练循环
if train_iter > 0 and detector_name != "ZF":
    optimizer = optim.Adam(model.parameters(), lr=startingLearningRate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_step_size, gamma=decay_factor)

    print('Training Start:', datetime.datetime.now())
    for i in range(train_iter):
        H, H_T, X, Y = generate_data(train_batch_size, K_orig, N_orig, snr_low, snr_high, mod)

        # 将 NumPy 数组转换为 PyTorch 张量并移动到设备
        H_tensor = torch.from_numpy(H).float().to(device)
        H_T_tensor = torch.from_numpy(H_T).float().to(device)
        X_tensor = torch.from_numpy(X).float().to(device)
        Y_tensor = torch.from_numpy(Y).float().to(device)

        optimizer.zero_grad()

        # 前向传播
        if detector_name == "ADMM":
            x_hat, x_result, layer_loss_list, _, _, _, _, _, _ = model(H_tensor, Y_tensor, H_T_tensor, X_tensor)
        elif detector_name == "Gated_ADMM":
            x_hat, x_result, layer_loss_list, _, _, _, _, _, _, _, _ = model(H_tensor, Y_tensor, H_T_tensor, X_tensor)

        # 计算总损失并反向传播
        total_loss = sum(layer_loss_list)
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if i % 1000 == 0:
            # 评估
            model.eval()
            with torch.no_grad():
                H_eval, H_T_eval, X_eval, Y_eval = generate_data(train_batch_size, K_orig, N_orig, snr_low, snr_high,
                                                                 mod)
                H_eval_tensor = torch.from_numpy(H_eval).float().to(device)
                H_T_eval_tensor = torch.from_numpy(H_T_eval).float().to(device)
                X_eval_tensor = torch.from_numpy(X_eval).float().to(device)
                Y_eval_tensor = torch.from_numpy(Y_eval).float().to(device)

                _, _, _, layer_ser_list, *_ = model(H_eval_tensor, Y_eval_tensor, H_T_eval_tensor, X_eval_tensor)
                eval_ser = sum(layer_ser_list) / len(layer_ser_list)  # 这里可能需要调整SER的计算方式

                print(f'Loss: {total_loss.item():3.6f} SER: {eval_ser.item():1.6f}')

            model.train()
            # 保存参数
            if detector_name == "ADMM":
                torch.save(model.state_dict(),
                           f'{save_file_name}/ADMM_Net_parameters_({K_orig}x{N_orig} {mod}QAM MIMO).pth')

    print('Training Finish:', datetime.datetime.now())

# 测试循环
print('Test Start:', datetime.datetime.now())
snrdb_list = np.linspace(snrdb_low, snrdb_high, num_snr)
snr_list = 10.0 ** (snrdb_list / 10.0)

for j in range(num_snr):
    total_sers = []
    total_times = []

    for jj in range(test_iter):
        H, H_T, X, Y = generate_data(test_batch_size, K_orig, N_orig, snr_list[j], snr_list[j], mod)
        H_tensor = torch.from_numpy(H).float().to(device)
        H_T_tensor = torch.from_numpy(H_T).float().to(device)
        X_tensor = torch.from_numpy(X).float().to(device)
        Y_tensor = torch.from_numpy(Y).float().to(device)

        tic = tm.time()
        with torch.no_grad():
            if detector_name == "ZF":
                H_T_H = torch.matmul(H_T_tensor, H_tensor)
                pinv_H = torch.matmul(torch.linalg.inv(H_T_H), H_T_tensor)
                x_hat = torch.matmul(pinv_H, Y_tensor)
                x_result = demod_projection(x_hat, mod)
                _, ser = loss(X_tensor, x_hat, x_result, K, mod)
                total_sers.append(ser.item())
            else:
                x_hat, x_result, _, layer_ser_list, *_ = model(H_tensor, Y_tensor, H_T_tensor, X_tensor)
                ser = sum(layer_ser_list) / len(layer_ser_list)
                total_sers.append(ser.item())
                # PyTorch 模型加载和保存时，需要用 model.state_dict()，然后用 torch.load 加载
        toc = tm.time()

        total_times.append((toc - tic) / test_batch_size)

    # 记录平均值
    sers[0][j] = np.mean(total_sers)
    times[0][j] = np.mean(total_times)

print('Test Finish:', datetime.datetime.now())
print('Test result')
print('snrdb_list : ', snrdb_list)
print('sers : ', sers)
print('times : ', times)