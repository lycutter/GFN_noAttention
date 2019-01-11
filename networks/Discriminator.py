import torch
import torch.nn as nn

from SAGFN import Self_Attn
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(0.2))
        self.conv_2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(0.2))
        self.conv_3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(0.2))
        self.conv_4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.LeakyReLU(0.2))
        self.conv_5 = nn.Sequential(nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1))



    def forward(self, x):

        out1 = self.conv_1(x)
        out2 = self.conv_2(out1)
        out3 = self.conv_3(out2)
        out4 = self.conv_4(out3)

        out = self.conv_5(out4)


        return out