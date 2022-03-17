import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from google.colab import drive
drive.mount('/content/drive', force_remount=True)
checkpoints = '/content/drive/MyDrive/455/birds/'


from google.colab import files
files.upload()
!pip install -q kaggle
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets list
!kaggle competitions download -c 'birds-22wi'
!mkdir ./drive/MyDrive/455
!unzip birds-22wi.zip -d ./drive/MyDrive/455


def get_birds_data():
    # Data augmentation transformations. Not for Testing!
    transform_train = transforms.Compose([
        transforms.Resize((300, 300)), # Takes images smaller than 300 and enlarges them
        transforms.RandomCrop(256, padding=4, padding_mode='edge'), # Take 256x256 crops from 300x300 images
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((256, 256)), # Takes images smaller than 512 and enlarges them
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.ImageFolder(root=checkpoints + 'train/', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    testset = torchvision.datasets.ImageFolder(root=checkpoints + 'test/', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    return {'train': trainloader, 'test': testloader}

data = get_birds_data()


class Darknet64(nn.Module):
    def __init__(self):
        super(Darknet64, self).__init__() # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        self.conv1 = nn.Conv2d(3, 96, 5, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(96)

        self.conv2 = nn.Conv2d(96, 128, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 192, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(192)

        self.conv4 = nn.Conv2d(192, 192, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(192)

        self.conv5 = nn.Conv2d(192, 256, 3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256, 2048)

        self.fc2 = nn.Linear(2048, 555)

    def forward(self, x):

        # Assume input is 256x256
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), kernel_size=4, stride=4) # 64x64x96
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), kernel_size=2, stride=2) # 32x32x128
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), kernel_size=2, stride=2) # 16x16x192
        x = F.relu(self.bn4(self.conv4(x)))                                        # 16x16x192
        x = F.relu(self.bn5(self.conv5(x)))                                        # 16x16x256

        # Global average pooling across each channel (Input could be 2x2x256, 4x4x256, 7x3x256, output would always be 256 length vector)
        x = F.adaptive_avg_pool2d(x, 1)                                            # 1x1x256
        x = torch.flatten(x, 1)                                                    # vector 256
        
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def train(net, dataloader, epochs=1, start_epoch=0, lr=0.01, momentum=0.9, decay=0.0005, 
          verbose=1, print_every=10, state=None, schedule={}, checkpoint_path=None):
    net.to(device)
    net.train()
    losses = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=decay)

    # Load previous training state
    if state:
        net.load_state_dict(state['net'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch']
        losses = state['losses']

    # Fast forward lr schedule through already trained epochs
    for epoch in range(start_epoch):
        if epoch in schedule:
            print ("Learning rate: %f"% schedule[epoch])
            for g in optimizer.param_groups:
                g['lr'] = schedule[epoch]

    for epoch in range(start_epoch, epochs):
        sum_loss = 0.0

        # Update learning rate when scheduled
        if epoch in schedule:
            print ("Learning rate: %f"% schedule[epoch])
            for g in optimizer.param_groups:
                g['lr'] = schedule[epoch]

        for i, batch in enumerate(dataloader, 0):
            inputs, labels = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # autograd magic, computes all the partial derivatives
            optimizer.step() # takes a step in gradient direction

            losses.append(loss.item())
            sum_loss += loss.item()
            
            if i % print_every == print_every-1:    # print every 10 mini-batches
                if verbose:
                    print('[%d, %5d] loss: %.3f' % (epoch, i + 1, sum_loss / print_every))
                sum_loss = 0.0
        if checkpoint_path:
            state = {'epoch': epoch+1, 'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'losses': losses}
            torch.save(state, checkpoint_path + 'checkpoint-%d.pkl'%(epoch+1))
    return losses

def accuracy(net, dataloader):
    net.to(device)
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch[0].to(device), batch[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct/total

def smooth(x, size):
    return np.convolve(x, np.ones(size)/size, mode='valid')


net = Darknet64()
state = torch.load(checkpoints + 'checkpoint-??.pkl')
losses = train(net, data['train'], epochs=70, schedule={0:.1, 4:.01, 9:.001, 20:.0001,30:0.001,44:0.0005,55:0.0002,65:0.0001}, checkpoint_path=checkpoints, state=state)


list = []

for element in data['train'].dataset.samples:
    if len(list) <= element[1]:
        list.insert(element[1], int(element[0][39:-37]))


def test(net, dataloader):
    net.to(device)
    with open(checkpoints + "testOutput.csv", "at") as f:
        with torch.no_grad():
            f.write("{},{}\n".format("path", "class"))
            for i, (images, labels) in enumerate(dataloader, 0):
                print(i)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                for j in range(len(predicted)):
                    fnames = "test/{}".format(str(dataloader.dataset.samples[i*64 + j][0])[40:])
                    f.write("{},{}\n".format(fnames, list[predicted[j]]))


net = Darknet64()
state = torch.load(checkpoints + 'checkpoint-30.pkl')
net.load_state_dict(state['net'])
test(net, data['test'])