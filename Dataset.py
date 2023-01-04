import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class DataSet(torch.utils.data.Dataset):
    def __init__(self, set, dataPath, dataName, width, height):
        super(DataSet, self).__init__()
        self.set = set
        self.path = dataPath
        self.name = dataName
        self.width = width
        self.height = height
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        img = Image.open(self.path + self.set + '\\Images\\' + self.name[idx] + '.png')
        img = np.asarray(img.resize((self.height, self.width), Image.NEAREST))
        img = np.array(img[:, :, 0])
        mask = Image.open(self.path + self.set + '\\Masks\\' + self.name[idx] + '.png')
        mask = np.asarray(mask.resize((self.height, self.width), Image.NEAREST))
        ###Image to Segmentation Map##############
        if self.set == 'Abdominal':
            maskDuo = np.where((mask[:, :, 0] == 255) &
                               (mask[:, :, 1] == 0) &
                               (mask[:, :, 2] == 0), 1, 0)

            maskEso = np.where((mask[:, :, 0] == 0) &
                               (mask[:, :, 1] == 255) &
                               (mask[:, :, 2] == 0), 1, 0)

            maskGal = np.where((mask[:, :, 0] == 0) &
                               (mask[:, :, 1] == 0) &
                               (mask[:, :, 2] == 255), 1, 0)

            maskLiv = np.where((mask[:, :, 0] == 255) &
                               (mask[:, :, 1] == 255) &
                               (mask[:, :, 2] == 0), 1, 0)

            maskLKi = np.where((mask[:, :, 0] == 255) &
                               (mask[:, :, 1] == 0) &
                               (mask[:, :, 2] == 255), 1, 0)

            maskPan = np.where((mask[:, :, 0] == 0) &
                               (mask[:, :, 1] == 255) &
                               (mask[:, :, 2] == 255), 1, 0)

            maskSpl = np.where((mask[:, :, 0] == 255) &
                               (mask[:, :, 1] == 255) &
                               (mask[:, :, 2] == 255), 1, 0)

            maskSto = np.where((mask[:, :, 0] == 128) &
                               (mask[:, :, 1] == 128) &
                               (mask[:, :, 2] == 128), 1, 0)

            maskBac = np.where((mask[:, :, 0] == 0) &
                               (mask[:, :, 1] == 0) &
                               (mask[:, :, 2] == 0), 1, 0)

            mask = np.dstack((maskDuo, maskEso, maskGal, maskLiv, maskLKi, maskPan, maskSpl, maskSto, maskBac))

        elif self.set == 'HeadNeck':
            maskBS = np.where((mask[:, :, 0] == 255) &
                               (mask[:, :, 1] == 0) &
                               (mask[:, :, 2] == 0), 1, 0)

            maskCh = np.where((mask[:, :, 0] == 0) &
                               (mask[:, :, 1] == 255) &
                               (mask[:, :, 2] == 0), 1, 0)

            maskMa = np.where((mask[:, :, 0] == 0) &
                               (mask[:, :, 1] == 0) &
                               (mask[:, :, 2] == 255), 1, 0)

            maskOL = np.where((mask[:, :, 0] == 255) &
                               (mask[:, :, 1] == 255) &
                               (mask[:, :, 2] == 0), 1, 0)

            maskOR = np.where((mask[:, :, 0] == 255) &
                               (mask[:, :, 1] == 0) &
                               (mask[:, :, 2] == 255), 1, 0)

            maskPL = np.where((mask[:, :, 0] == 0) &
                               (mask[:, :, 1] == 255) &
                               (mask[:, :, 2] == 255), 1, 0)

            maskPR = np.where((mask[:, :, 0] == 255) &
                               (mask[:, :, 1] == 255) &
                               (mask[:, :, 2] == 255), 1, 0)

            maskSL = np.where((mask[:, :, 0] == 128) &
                               (mask[:, :, 1] == 128) &
                               (mask[:, :, 2] == 128), 1, 0)

            maskSR = np.where((mask[:, :, 0] == 64) &
                              (mask[:, :, 1] == 128) &
                              (mask[:, :, 2] == 255), 1, 0)

            maskBac = np.where((mask[:, :, 0] == 0) &
                               (mask[:, :, 1] == 0) &
                               (mask[:, :, 2] == 0), 1, 0)

            mask = np.dstack((maskBS, maskCh, maskMa, maskOL, maskOR, maskPL, maskPR, maskSL, maskSR, maskBac))

        img = self.transform(img)
        mask = self.transform(mask)
        return img, mask
    
