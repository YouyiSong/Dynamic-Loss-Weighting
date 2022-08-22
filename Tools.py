import numpy as np
import os
import torch
import random
import math


def DeviceInitialization(GPUNum):
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        device = torch.device(GPUNum)
    else:
        device = torch.device('cpu')

    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)
    return device


def DataReading(path, set, fracTrain):
    trainIdx = []
    testIdx = []
    TrainIdx = []
    TestIdx = []

    if set == 'Abdominal':
        data = np.genfromtxt(path + set + '\\CSVs\\' + 'BTCV.txt', dtype=str)
        shuffleIdx = np.arange(len(data))
        shuffleRng = np.random.RandomState(2021)
        shuffleRng.shuffle(shuffleIdx)
        data = data[shuffleIdx]

        TrainNum = math.ceil(fracTrain * len(data) / 100)
        train = data[:TrainNum]
        test = data[TrainNum:]

        for ii in range(len(train)):
            trainIdx.append('BTCV_' + train[ii])
        for ii in range(len(test)):
            testIdx.append('BTCV_' + test[ii])

        for ii in range(len(trainIdx)):
            data = trainIdx[ii]
            file = path + set + '\\CSVs\\Slice\\' + data + '.txt'
            if os.path.isfile(file):
                dataSlice = np.genfromtxt(file, dtype=str)
                for jj in range(len(dataSlice)):
                    temp = data + '_' + dataSlice[jj]
                    TrainIdx.append(temp)

        for ii in range(len(testIdx)):
            data = testIdx[ii]
            file = path + set + '\\CSVs\\Slice\\' + data + '.txt'
            if os.path.isfile(file):
                dataSlice = np.genfromtxt(file, dtype=str)
                for jj in range(len(dataSlice)):
                    temp = data + '_' + dataSlice[jj]
                    TestIdx.append(temp)

        data = np.genfromtxt(path + set + '\\CSVs\\' + 'TCIA.txt', dtype=str)
        shuffleIdx = np.arange(len(data))
        shuffleRng = np.random.RandomState(2021)
        shuffleRng.shuffle(shuffleIdx)
        data = data[shuffleIdx]
        TrainNum = math.ceil(fracTrain * len(data) / 100)
        train = data[:TrainNum]
        test = data[TrainNum:]

        for ii in range(len(train)):
            trainIdx.append('TCIA_' + train[ii])
        for ii in range(len(test)):
            testIdx.append('TCIA_' + test[ii])

        for ii in range(len(trainIdx)):
            data = trainIdx[ii]
            file = path + set + '\\CSVs\\Slice\\' + data + '.txt'
            if os.path.isfile(file):
                dataSlice = np.genfromtxt(file, dtype=str)
                for jj in range(len(dataSlice)):
                    temp = data + '_' + dataSlice[jj]
                    TrainIdx.append(temp)

        for ii in range(len(testIdx)):
            data = testIdx[ii]
            file = path + set + '\\CSVs\\Slice\\' + data + '.txt'
            if os.path.isfile(file):
                dataSlice = np.genfromtxt(file, dtype=str)
                for jj in range(len(dataSlice)):
                    temp = data + '_' + dataSlice[jj]
                    TestIdx.append(temp)

    elif set == 'HeadNeck':
        data = np.genfromtxt(path + set + '\\CSVs\\' + 'HeadNeck.txt', dtype=str)
        shuffleIdx = np.arange(len(data))
        shuffleRng = np.random.RandomState(2021)
        shuffleRng.shuffle(shuffleIdx)
        data = data[shuffleIdx]

        TrainNum = math.ceil(fracTrain * len(data) / 100)
        train = data[:TrainNum]
        test = data[TrainNum:]

        for ii in range(len(train)):
            trainIdx.append(train[ii])
        for ii in range(len(test)):
            testIdx.append(test[ii])

        for ii in range(len(trainIdx)):
            data = trainIdx[ii]
            file = path + set + '\\CSVs\\Slice\\' + data + '.txt'
            if os.path.isfile(file):
                dataSlice = np.genfromtxt(file, dtype=str)
                for jj in range(len(dataSlice)):
                    temp = data + '_' + dataSlice[jj]
                    TrainIdx.append(temp)

        for ii in range(len(testIdx)):
            data = testIdx[ii]
            file = path + set + '\\CSVs\\Slice\\' + data + '.txt'
            if os.path.isfile(file):
                dataSlice = np.genfromtxt(file, dtype=str)
                for jj in range(len(dataSlice)):
                    temp = data + '_' + dataSlice[jj]
                    TestIdx.append(temp)

    return TrainIdx, TestIdx


class Sampler:
    def __init__(self, device, model, img_shape, sample_size, max_len, step, lr):
        super().__init__()
        self.device = device
        self.model = model
        self.img_shape = img_shape
        self.sample_size = sample_size
        self.max_len = max_len
        self.step = step
        self.lr = lr
        self.examples = [(torch.rand((1,) + img_shape) * 2 - 1) for _ in range(self.sample_size.item())]

    def sample_new_exmps(self):
        # Choose 95% of the batch from the buffer, 5% generate from scratch
        n_new = np.random.binomial(self.sample_size.item(), 0.05)
        rand_imgs = torch.rand((n_new,) + self.img_shape) * 2 - 1
        old_imgs = torch.cat(random.choices(self.examples, k=self.sample_size.item() - n_new), dim=0)
        inp_imgs = torch.cat([rand_imgs, old_imgs], dim=0).detach().to(self.device)
        inp_imgs = Sampler.generate_samples(self.model.to(self.device), inp_imgs, steps=self.step, step_size=self.lr)
        # Add new images to the buffer and remove old ones if needed
        self.examples = list(inp_imgs.to(torch.device('cpu')).chunk(self.sample_size.item(), dim=0)) + self.examples
        self.examples = self.examples[:self.max_len]
        return inp_imgs

    def generate_samples(model, inp_imgs, steps, step_size):
        for p in model.parameters():
            p.requires_grad = False
        inp_imgs.requires_grad = True
        noise = torch.randn(inp_imgs.shape, device=inp_imgs.device)
        for idx in range(steps):
            out_imgs = -model(inp_imgs)
            out_imgs.sum().backward()
            inp_imgs.grad.data.clamp_(-0.03, 0.03)
            noise.normal_(0, 0.005)
            inp_imgs.data.add_(-step_size * inp_imgs.grad.data + noise.data)
            inp_imgs.grad.detach_()
            inp_imgs.grad.zero_()
            inp_imgs.data.clamp_(min=-1.0, max=1.0)

        inp_imgs.requires_grad = False
        inp_imgs.detach()
        for p in model.parameters():
            p.requires_grad = True

        return inp_imgs


def Sampling(model, inp_imgs, step, step_size):
    inp_imgs.requires_grad = True
    for idx in range(step):
        inp_imgs.data.clamp_(min=-1.0, max=1.0)
        out_imgs = -model(inp_imgs)
        out_imgs.sum().backward()
        inp_imgs.grad.data.clamp_(-0.1, 0.1)
        inp_imgs.data.add_(-step_size * inp_imgs.grad.data)
        inp_imgs.grad.detach_()
        inp_imgs.grad.zero_()
        inp_imgs.data.clamp_(min=-1.0, max=1.0)
    inp_imgs.requires_grad = False
    inp_imgs.detach()

    return inp_imgs