import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Gaussian_Conv(nn.Module):

    def __init__(self):
        super(Gaussian_Conv, self).__init__()
        Gkernel = gaussian_2d_kernel(5, 1)
        Gkernel = torch.FloatTensor(Gkernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=Gkernel, requires_grad=False).cuda()


    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=4)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding=4)
        x3 = F.conv2d(x3.unsqueeze(1), self.weight, padding=4)
        x = torch.cat([x1, x2, x3], dim=1)
        return x



def gaussian_2d_kernel(kernel_size=5, sigma=1):

    kernel = np.zeros([kernel_size,kernel_size])
    center = kernel_size//2

    if sigma == 0:
        sigma = ((kernel_size-1)*0.5 - 1)*0.3 + 0.8

    s = 2*(sigma**2)
    sum_val = 0
    for i in range(0,kernel_size):
        for j in range(0,kernel_size):
            x = i-center
            y = j-center
            kernel[i,j] = np.exp(-(x**2+y**2) / s)
            sum_val += kernel[i,j]
            #/(np.pi * s)
    sum_val = 1/sum_val
    return kernel*sum_val