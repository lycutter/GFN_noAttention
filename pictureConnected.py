import os
from PIL import Image

# import pyautogui
# import re


dirs = []


dir0 = 'D:/pythonWorkplace/GFN-master/datasets/LR-GOPRO/Validation_new/Results/'
dir1 = 'D:/pythonWorkplace/GFN-GAN/datasets/LR-GOPRO/Validation_new/MSRN-GAN-L1/'
dir2 = 'D:/pythonWorkplace/GFN-GAN/datasets/LR-GOPRO/Validation_new/GFN_orgin-GAN-L1/'
dir3 = 'D:/pythonWorkplace/GFN-GAN/datasets/LR-GOPRO/Validation_new/MSRN-GAN-C-L1-Gate/'
dir4 = 'D:/pythonWorkplace/GFN-GAN/datasets/LR-GOPRO/Validation_new/MSRN-GAN-C-L1-noGate/'
dir5 = 'D:/pythonWorkplace/GFN-GAN/datasets/LR-GOPRO/Validation_new/MSRN-GAN-EPAT/'
dir6 = 'D:/pythonWorkplace/GFN-GAN/datasets/LR-GOPRO/Validation_new/HR/'
dir_1 = 'D:/pythonWorkplace/GFN-GAN/datasets/LR-GOPRO/Validation_4/LR/'
dir7 = 'D:/pythonWorkplace/GFN-master/datasets/LR-GOPRO/Validation_new/GFN_orgin_GAN-L1-noAttention/'

dirs.append(dir_1)
dirs.append(dir0)
dirs.append(dir1)
dirs.append(dir2)
dirs.append(dir3)
dirs.append(dir4)
dirs.append(dir5)
dirs.append(dir6)
dirs.append(dir7)

dirPicList_1 = []
dirPicList0 = []
dirPicList1 = []
dirPicList2 = []
dirPicList3 = []
dirPicList4 = []
dirPicList5 = []
dirPicList6 = []
dirPicList7 = []


list1 = os.listdir(dir1)
listhr = os.listdir(dir6)


for i in range(688):
    dirPicList0.append(dir0 + list1[i])
    dirPicList1.append(dir1 + list1[i])
    dirPicList2.append(dir2 + list1[i])
    dirPicList3.append(dir3 + list1[i])
    dirPicList4.append(dir4 + list1[i])
    dirPicList5.append(dir5 + list1[i])
    dirPicList6.append(dir6 + listhr[i])
    dirPicList_1.append(dir_1 + list1[i])
    dirPicList7.append(dir7 + list1[i])








toImage = Image.new('RGBA', (128*8, 128*8))

for i in range(688):

    # 第一行
    pic_fole_head_0 = Image.open(dirPicList0[i])
    toImage.paste(pic_fole_head_0, (0, 0))
    pic_fole_head_1 = Image.open(dirPicList1[i])
    toImage.paste(pic_fole_head_1, (128, 0))
    pic_fole_head_2 = Image.open(dirPicList2[i])
    toImage.paste(pic_fole_head_2, (256, 0))
    pic_fole_head_3 = Image.open(dirPicList3[i])
    toImage.paste(pic_fole_head_3, (384, 0))
    pic_fole_head_4 = Image.open(dirPicList4[i])
    toImage.paste(pic_fole_head_4, (512, 0))
    pic_fole_head_5 = Image.open(dirPicList5[i])
    toImage.paste(pic_fole_head_5, (640, 0))
    pic_fole_head_6 = Image.open(dirPicList6[i])
    toImage.paste(pic_fole_head_6, (768, 0))
    pic_fole_head_7 = Image.open(dirPicList_1[i])
    toImage.paste(pic_fole_head_7, (896, 0))

    # 第二行
    pic_fole_head_0 = Image.open(dirPicList0[i+1])
    toImage.paste(pic_fole_head_0, (0, 128))
    pic_fole_head_1 = Image.open(dirPicList1[i+1])
    toImage.paste(pic_fole_head_1, (128, 128))
    pic_fole_head_2 = Image.open(dirPicList2[i+1])
    toImage.paste(pic_fole_head_2, (256, 128))
    pic_fole_head_3 = Image.open(dirPicList3[i+1])
    toImage.paste(pic_fole_head_3, (384, 128))
    pic_fole_head_4 = Image.open(dirPicList4[i+1])
    toImage.paste(pic_fole_head_4, (512, 128))
    pic_fole_head_5 = Image.open(dirPicList5[i+1])
    toImage.paste(pic_fole_head_5, (640, 128))
    pic_fole_head_6 = Image.open(dirPicList6[i+1])
    toImage.paste(pic_fole_head_6, (768, 128))
    pic_fole_head_7 = Image.open(dirPicList_1[i+1])
    toImage.paste(pic_fole_head_7, (896, 128))

    # 第三行
    pic_fole_head_0 = Image.open(dirPicList0[i+2])
    toImage.paste(pic_fole_head_0, (0, 256))
    pic_fole_head_1 = Image.open(dirPicList1[i+2])
    toImage.paste(pic_fole_head_1, (128, 256))
    pic_fole_head_2 = Image.open(dirPicList2[i+2])
    toImage.paste(pic_fole_head_2, (256, 256))
    pic_fole_head_3 = Image.open(dirPicList3[i+2])
    toImage.paste(pic_fole_head_3, (384, 256))
    pic_fole_head_4 = Image.open(dirPicList4[i+2])
    toImage.paste(pic_fole_head_4, (512, 256))
    pic_fole_head_5 = Image.open(dirPicList5[i+2])
    toImage.paste(pic_fole_head_5, (640, 256))
    pic_fole_head_6 = Image.open(dirPicList6[i+2])
    toImage.paste(pic_fole_head_6, (768, 256))
    pic_fole_head_7 = Image.open(dirPicList_1[i+2])
    toImage.paste(pic_fole_head_7, (896, 256))

    #第四行
    pic_fole_head_0 = Image.open(dirPicList0[i+3])
    toImage.paste(pic_fole_head_0, (0, 384))
    pic_fole_head_1 = Image.open(dirPicList1[i+3])
    toImage.paste(pic_fole_head_1, (128, 384))
    pic_fole_head_2 = Image.open(dirPicList2[i+3])
    toImage.paste(pic_fole_head_2, (256, 384))
    pic_fole_head_3 = Image.open(dirPicList3[i+3])
    toImage.paste(pic_fole_head_3, (384, 384))
    pic_fole_head_4 = Image.open(dirPicList4[i+3])
    toImage.paste(pic_fole_head_4, (512, 384))
    pic_fole_head_5 = Image.open(dirPicList5[i+3])
    toImage.paste(pic_fole_head_5, (640, 384))
    pic_fole_head_6 = Image.open(dirPicList6[i+3])
    toImage.paste(pic_fole_head_6, (768, 384))
    pic_fole_head_7 = Image.open(dirPicList_1[i+3])
    toImage.paste(pic_fole_head_7, (896, 384))

    # 第五行
    pic_fole_head_0 = Image.open(dirPicList0[i+4])
    toImage.paste(pic_fole_head_0, (0, 512))
    pic_fole_head_1 = Image.open(dirPicList1[i+4])
    toImage.paste(pic_fole_head_1, (128, 512))
    pic_fole_head_2 = Image.open(dirPicList2[i+4])
    toImage.paste(pic_fole_head_2, (256, 512))
    pic_fole_head_3 = Image.open(dirPicList3[i+4])
    toImage.paste(pic_fole_head_3, (384, 512))
    pic_fole_head_4 = Image.open(dirPicList4[i+4])
    toImage.paste(pic_fole_head_4, (512, 512))
    pic_fole_head_5 = Image.open(dirPicList5[i+4])
    toImage.paste(pic_fole_head_5, (640, 512))
    pic_fole_head_6 = Image.open(dirPicList6[i+4])
    toImage.paste(pic_fole_head_6, (768, 512))
    pic_fole_head_7 = Image.open(dirPicList_1[i+4])
    toImage.paste(pic_fole_head_7, (896, 512))

    # 第六行
    pic_fole_head_0 = Image.open(dirPicList0[i+5])
    toImage.paste(pic_fole_head_0, (0, 640))
    pic_fole_head_1 = Image.open(dirPicList1[i+5])
    toImage.paste(pic_fole_head_1, (128, 640))
    pic_fole_head_2 = Image.open(dirPicList2[i+5])
    toImage.paste(pic_fole_head_2, (256, 640))
    pic_fole_head_3 = Image.open(dirPicList3[i+5])
    toImage.paste(pic_fole_head_3, (384, 640))
    pic_fole_head_4 = Image.open(dirPicList4[i+5])
    toImage.paste(pic_fole_head_4, (512, 640))
    pic_fole_head_5 = Image.open(dirPicList5[i+5])
    toImage.paste(pic_fole_head_5, (640, 640))
    pic_fole_head_6 = Image.open(dirPicList6[i+5])
    toImage.paste(pic_fole_head_6, (768, 640))
    pic_fole_head_7 = Image.open(dirPicList_1[i+5])
    toImage.paste(pic_fole_head_7, (896, 640))

    # 第七行
    pic_fole_head_0 = Image.open(dirPicList0[i+6])
    toImage.paste(pic_fole_head_0, (0, 768))
    pic_fole_head_1 = Image.open(dirPicList1[i+6])
    toImage.paste(pic_fole_head_1, (128, 768))
    pic_fole_head_2 = Image.open(dirPicList2[i+6])
    toImage.paste(pic_fole_head_2, (256, 768))
    pic_fole_head_3 = Image.open(dirPicList3[i+6])
    toImage.paste(pic_fole_head_3, (384, 768))
    pic_fole_head_4 = Image.open(dirPicList4[i+6])
    toImage.paste(pic_fole_head_4, (512, 768))
    pic_fole_head_5 = Image.open(dirPicList5[i+6])
    toImage.paste(pic_fole_head_5, (640, 768))
    pic_fole_head_6 = Image.open(dirPicList6[i+6])
    toImage.paste(pic_fole_head_6, (768, 768))
    pic_fole_head_7 = Image.open(dirPicList_1[i+6])
    toImage.paste(pic_fole_head_7, (896, 768))

    # 第八行
    pic_fole_head_0 = Image.open(dirPicList0[i+7])
    toImage.paste(pic_fole_head_0, (0, 896))
    pic_fole_head_1 = Image.open(dirPicList1[i+7])
    toImage.paste(pic_fole_head_1, (128, 896))
    pic_fole_head_2 = Image.open(dirPicList2[i+7])
    toImage.paste(pic_fole_head_2, (256, 896))
    pic_fole_head_3 = Image.open(dirPicList3[i+7])
    toImage.paste(pic_fole_head_3, (384, 896))
    pic_fole_head_4 = Image.open(dirPicList4[i+7])
    toImage.paste(pic_fole_head_4, (512, 896))
    pic_fole_head_5 = Image.open(dirPicList5[i+7])
    toImage.paste(pic_fole_head_5, (640, 896))
    pic_fole_head_6 = Image.open(dirPicList6[i+7])
    toImage.paste(pic_fole_head_6, (768, 896))
    pic_fole_head_7 = Image.open(dirPicList_1[i+7])
    toImage.paste(pic_fole_head_7, (896, 896))

    i = i + 8

    toImage = toImage.convert('RGB')
    path = 'D:/pythonWorkplace/GFN-GAN/datasets/LR-GOPRO/Validation_new/picConnected/' + str(int(i/8)) + '.jpg'
    toImage.save(path)





