import torch
import time
from Tools import DeviceInitialization
from Tools import DataReading
from torch.utils.data import DataLoader
from Dataset import DataSet
from Model import UNet
from Model import ResNet
from Model import Inception
from Model import Encoder
from Loss import Loss
from Tools import Sampling


#############################################################################
set = 'Abdominal' ## 'Abdominal' or 'HeadNeck'
lossType = 'Dice' ##'Dice' or 'CE'
modelName = set + '_Our' + '_UNet_' + lossType
batch_size = 16  ## 16 for 'Abdominal' and 2 for 'HeadNeck'
class_num = 9    ## 9 for 'Abdominal' and 10 for 'HeadNeck'
############################################################################
epoch_num = 20 ## 20 for 'Abdominal' and 40 for 'HeadNeck'
learning_rate = 3e-4
path = 'D:\\LossIncentive\\Data\\'
modelPath = 'D:\\LossIncentive\\Model\\'
fracTrain = 50

device = DeviceInitialization('cuda:0')
TrainIdx, TestIdx = DataReading(path=path, set=set, fracTrain=fracTrain)
trainSet = DataSet(dataPath=path, set=set, dataName=TrainIdx, height=128, width=128)
testSet = DataSet(dataPath=path, set=set, dataName=TestIdx, height=128, width=128)
TrainSet = torch.utils.data.DataLoader(dataset=trainSet, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
TestSet = torch.utils.data.DataLoader(dataset=testSet, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

###################################################
Net = UNet(obj_num=class_num)    ########
#Net = ResNet(obj_num=class_num)  #########
#Net = Inception(obj_num=class_num)  ######
###################################################
Net.to(device)
optim = torch.optim.Adam(Net.parameters(), lr=learning_rate)
criterion = Loss(mode=lossType)

CoderReg = Encoder(num_obj=class_num)
Saved = torch.load(modelPath + 'Encoder_' + set + '_PreTrain.pth')
CoderReg.load_state_dict(Saved['Net'])
CoderReg.to(device)
for p in CoderReg.parameters():
    p.requires_grad = False

Coder = Encoder(num_obj=class_num)
Coder.load_state_dict(Saved['Net'])
Coder.to(device)
optimCoder = torch.optim.Adam(Coder.parameters(), lr=learning_rate)
optimCoder.load_state_dict(Saved['optim'])

torch.cuda.synchronize()
start_time = time.time()
IterNum = 0
for epoch in range(epoch_num):
    Net.train()
    tempLoss = 0
    tempEdis = 0
    for idx, (images, targets) in enumerate(TrainSet, 0):
        images = images.to(device)
        targets = targets.to(device)

        outputs = Net(images)
        _, segIdx = torch.max(outputs.detach(), dim=1)
        fakes = torch.zeros_like(targets)
        for ii in range(class_num):
            fakes[:, ii] = torch.where(segIdx == ii, torch.ones_like(segIdx), torch.zeros_like(segIdx))

        fakes = 2 * (fakes - 0.5)
        lossCoder = -(Coder(2.0 * (targets - 0.5)) - Coder(fakes)).mean() + 0.1 * (Coder(2.0 * (targets - 0.5)) - Coder(fakes) ** 2).mean()
        lossCoderReg = ((Coder(fakes) - CoderReg(fakes)) ** 2).mean()
        loss = 4.0 * lossCoderReg + lossCoder
        optimCoder.zero_grad()
        loss.backward()
        optimCoder.step()

        fakes = Sampling(Coder, fakes, 10, 10)
        fakes = fakes / 2 + 0.5
        lossReal = criterion(outputs, targets)
        lossFake = criterion(fakes, targets)
        tempEdis += lossReal.detach().sum() - lossFake.detach().sum()

        weight = abs(lossReal.detach() - lossFake)
        for ii in range(weight.size(0)):
            temp = weight[ii]
            if max(temp) > 0:
                temp = 0.1 * (temp - min(temp)) / max(temp) + 0.9
                weight[ii] = temp / temp.sum()
            else:
                weight[ii] = 1.0 / class_num

        loss = (weight * lossReal).sum()
        optim.zero_grad()
        loss.backward()
        optim.step()
        tempLoss += loss

    IterNum += (idx + 1)
    print("Epoch:%02d  ||  Iteration:%04d  ||  Loss:%.4f  ||  Dis:%.4f  ||  Time elapsed:%.2f(min)"
          % (epoch + 1, IterNum, tempLoss / (idx + 1), tempEdis / (idx + 1), (time.time() - start_time) / 60))

torch.save({'Net': Net.state_dict(), 'optimNet': optim.state_dict(), 'Coder': Coder.state_dict(), 'optimCoder': optimCoder.state_dict()}, modelPath + modelName + '.pth')