import torch
import numpy as np
import torch.nn as nn

pi = np.pi


#gradient penalty function
def gradient_penalty(critic, real, fake, device="cpu"):
    [bat, cha, row, col] = real.shape
    epsilon = torch.rand((bat, 1, 1, 1)).repeat(1, cha, row, col).to(device)
    interpolated_imges = real * epsilon + fake * (1 - epsilon)

    #calculate critic scores
    mixed_scores = critic(interpolated_imges)

    gradient = torch.autograd.grad(
        inputs=interpolated_imges,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


#specturm function
def transmission_by_specturm(real_size, object_matrix, z, lam, device="cpu"):
    #光的角谱传播函数，real_size是光源的几何尺寸，object_matrix是光源矩阵
    #z是传播距离，lam是光波长，像面尺寸与物面尺寸大小一样
    #object的最小采样间隔大于0.061微米
    #输入矩阵最好是2的n次方这样fft算的最快
    [bat, cha, row, col] = object_matrix.shape
    df = 1/real_size
    k = df*np.linspace(-row/2, (row/2-1), row)
    [fx, fy] = np.meshgrid(k, k)
    lx = 1/(np.sqrt(1+(2*z/real_size)**2)*lam)
    ly = lx
    p = 1-lam**2*(fx**2+fy**2)
    NA = (fx > -lx)*(fx < lx)*(fy > -ly)*(fy < ly)*1
    q1 = np.fft.fftshift(NA*np.exp(2*pi*1j*z/lam*np.sqrt(p)))
    q = torch.tensor(q1.reshape([1, cha, row, col]), device=device)
    result = torch.fft.ifft2(torch.fft.fft2(object_matrix)*q)
    return result


def one_target_img(target, M):
    k = np.linspace(1, 128, 128)
    [x, y] = np.meshgrid(k, k)
    row_index = 0     #the counts of target region
    col_index = 0
    judge = [3, 4, 3]
    region_size = 128/(2**3)     #the size of target region
    targets = target + 1
    while targets > judge[row_index]:
        targets = targets - judge[row_index]
        row_index = row_index + 1
    col_index = int(targets - 1)
    interval_col = int((128 - judge[row_index] * region_size) / (2 * judge[row_index]))
    interval_row = int((128 - 3 * region_size) / 6)
    interval_matrix = [[2, 1, 1, 0, 0], [3, 1, 1, 1, 0], [2, 1, 1, 0, 0]]
    interval_matrix_row = [2, 1, 1, 0, 0]
    be_col = np.sum(interval_matrix[row_index][0:col_index + 1]) * interval_col + col_index * region_size
    en_col = be_col + region_size
    be_row = np.sum(interval_matrix_row[0:row_index + 1]) * interval_row + row_index * region_size
    en_row = be_row + region_size
    targetimg = (x >= be_col) * (x <= en_col) * (y >= be_row) * (y <= en_row) * 1.0
    pad_num = int((M - 128) / 2)
    targetimg = np.pad(targetimg, ((pad_num, pad_num), (pad_num, pad_num)))
    return targetimg


def target_imgs(targets, M, device="cpu"):
    bat = targets.shape[0]
    targetimgs = torch.ones(bat, 1, M, M, device=device, dtype=torch.float64)
    for i in range(bat):
        kk = torch.tensor(one_target_img(targets[i].item(), M), device=device)
        targetimgs[i][0] = targetimgs[i][0]*kk
    return targetimgs

def area_energy(inputs, device="cpu"):
    [bat, cha, row, col] = inputs.shape
    tars = torch.linspace(0, 9, 10, device=device, dtype=torch.float64)
    TargetsImgs = target_imgs(tars, row, device=device)
    tem_inp = inputs[0].reshape((1, 1, row, row))
    result = torch.sum(torch.sum(tem_inp * TargetsImgs, dim=2), dim=2).view(-1)
    for i in range(bat-1):
        tem_inp = inputs[i+1].reshape((1, 1, row, row))
        tem = torch.sum(torch.sum(tem_inp*TargetsImgs, dim=2), dim=2).view(-1)
        result = torch.vstack((result, tem))
    return result

