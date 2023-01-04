import torch
import time
from Tools import DeviceInitialization
from Tools import DataReading
from torch.utils.data import DataLoader
from Dataset import DataSet
from Model import Encoder
from Tools import Sampler


set = 'Abdominal' ## 'Abdominal' or 'HeadNeck'
modelName = 'Encoder_' + set + '_'
epoch_num = 10 ## 10 for 'Abdominal' and 20 for 'HeadNeck'
batch_size = torch.tensor([16]) ## 16 for 'Abdominal' and 2 for 'HeadNeck'
img_size = 128
num_obj = 9 ## 9 for 'Abdominal' and 10 for 'HeadNeck'
learning_rate = 3e-4
reg_weight = 0.0001
path = 'D:\\LossIncentive\\Data\\'
modelPath = 'D:\\LossIncentive\\Model\\'
fracTrain = 80

device = DeviceInitialization('cuda:0')
TrainIdx, TestIdx = DataReading(path=path, set=set, fracTrain=fracTrain)
trainSet = DataSet(dataPath=path, set=set, dataName=TrainIdx, height=128, width=128)
testSet = DataSet(dataPath=path, set=set, dataName=TestIdx, height=128, width=128)
TrainSet = torch.utils.data.DataLoader(dataset=trainSet, batch_size=batch_size.item(), shuffle=True, num_workers=0, pin_memory=True)
TestSet = torch.utils.data.DataLoader(dataset=testSet, batch_size=batch_size.item(), shuffle=False, num_workers=0, pin_memory=True)

Coder = Encoder(num_obj=num_obj)
Coder.to(device)
optim = torch.optim.Adam(Coder.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optim, 1, gamma=0.995)
Sampler = Sampler(device, Coder, (num_obj, img_size, img_size), batch_size, 8192, 40, 10)

torch.cuda.synchronize()
start_time = time.time()
IterNum = 0
tempLossReal = 0
tempLossFake = 0
for epoch in range(epoch_num):
    torch.set_printoptions(precision=4, sci_mode=False)
    for idx, (_, real_imgs) in enumerate(TrainSet, 0):
        ### Getting real & fakes images
        batch_size[0] = real_imgs.size(0)
        real_imgs = real_imgs.to(device)
        real_imgs = 2.0 * (real_imgs - 0.5)
        real_imgs += 0.005 * torch.randn_like(real_imgs)
        real_imgs = real_imgs.clamp(min=-1.0, max=1.0)
        fake_imgs = Sampler.sample_new_exmps()
        imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        ### Updating Encoder
        real_out, fake_out = Coder(imgs).chunk(2, dim=0)
        div = fake_out.mean() - real_out.mean()
        reg = (real_out ** 2).mean() + (fake_out ** 2).mean()
        loss = div + reg_weight * reg
        optim.zero_grad()
        loss.backward()
        optim.step()
        ### Outputing training information
        tempLossReal += real_out.detach().mean()
        tempLossFake += fake_out.detach().mean()
        IterNum += 1

        if IterNum % 100 == 0:
            torch.cuda.synchronize()
            print("Epoch:%02d  ||  Iteration:%04d  ||  RealLoss:%.4f  ||  FakeLoss:%.4f  ||  Time elapsed:%.2f(min)"
                  % (epoch + 1, IterNum, tempLossReal / 100, tempLossFake / 100, (time.time() - start_time) / 60))
            scheduler.step()
            tempLossReal = 0
            tempLossFake = 0

torch.save({'Net': Coder.state_dict(), 'optim': optim.state_dict()}, modelPath + modelName + 'PreTrain.pth')
