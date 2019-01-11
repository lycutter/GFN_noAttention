import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, Normalize

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)

        self.AB_paths = sorted(make_dataset(self.dir_AB))

        # assert(opt.resize_or_crop == 'resize_and_crop')

        # transform_list = [transforms.ToTensor(),
        #                   transforms.Normalize((0.5, 0.5, 0.5),
        #                                        (0.5, 0.5, 0.5))]
        transform_list = [ToTensor()]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # AB = Image.open(AB_path)
        # AB = AB.resize((self.opt.loadSizeX * 2, self.opt.loadSizeY), Image.BICUBIC)
        AB = self.transform(AB)

        # w_total = AB.size(2)
        # w = int(w_total / 2)
        # h = AB.size(1)
        # w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        # h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))
        #
        # A = AB[:, h_offset:h_offset + self.opt.fineSize, # blurred img
        #        w_offset:w_offset + self.opt.fineSize]
        # B = AB[:, h_offset:h_offset + self.opt.fineSize, # sharp img
        #        w + w_offset:w + w_offset + self.opt.fineSize]

        A = AB[:, 0:128, 0:128]
        B = AB[:, 0:128, 128:256]

        lr_tranform = train_lr_transform(self.opt.fineSize, 4)


        D = A.clone()
        A = lr_tranform(A)
        C = lr_tranform(B)



        if (not self.opt.no_flip) and random.random() < 0.5:  # 翻转图片，扩展数据量
            # idx = [i for i in range(A.size(2) - 1, -1, -1)]
            # idx = torch.LongTensor(idx)
            # A = A.index_select(2, idx)
            # B = B.index_select(2, idx)
            idx_A = [i for i in range(A.size(2) - 1, -1, -1)]
            idx_B = [i for i in range(B.size(2) - 1, -1, -1)]
            idx_C = [i for i in range(C.size(2) - 1, -1, -1)]
            idx_D = [i for i in range(D.size(2) - 1, -1, -1)]

            idx_A = torch.LongTensor(idx_A)
            idx_B = torch.LongTensor(idx_B)
            idx_C = torch.LongTensor(idx_C)
            idx_D = torch.LongTensor(idx_D)

            A = A.index_select(2, idx_A)
            B = B.index_select(2, idx_B)
            C = C.index_select(2, idx_C)
            D = D.index_select(2, idx_D)


        return {'LR_Blur': A, 'HR_Sharp': B, 'LR_Sharp': C, 'HR_Blur': D} # A是LR_Blur， B是HR_Sharp, C是LR_Sharp

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(128 // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])

def train_lr_transformx16(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(64 // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])

def train_lr_transformx8(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(32 // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])
