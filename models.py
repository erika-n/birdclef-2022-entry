

import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T


class BirdConv1d(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=200):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        #print('start of foward', x.size())
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        #print('after pool4', x.size())
        x = F.avg_pool1d(x, x.shape[-1])
        #print('after avg_pool1d', x.size())
        x = x.permute(0, 2, 1)
        #print('after permute', x.size())
        x = self.fc1(x)
        #print('after fc1', x.size())
  
        return F.log_softmax(x, dim=2)



class BirdConv2d(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=2, n_channel=20):
        super().__init__()

        n_fft = 800
        win_length = None
        hop_length = 100

        # define transformation
        self.spectrogram = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
        )

        self.conv1 = nn.Conv2d(n_input, n_channel, kernel_size=5, stride=stride)
        self.bn1 = nn.BatchNorm2d(n_channel)
        self.pool1 = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(n_channel, n_channel, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(n_channel)
        self.pool2 = nn.MaxPool2d(3)
        self.conv3 = nn.Conv2d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(2 * n_channel)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(2 * n_channel)
        self.pool4 = nn.MaxPool2d(2)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(840, 100)
        self.fc2 = nn.Linear(100, n_output)

    def forward(self, x):
        #print('start of forward', x.size())
        x = self.spectrogram(x)
        #print('after spectrogram', x.size())
        x = self.conv1(x)
        #print('after conv1', x.size())
        x = F.relu(self.bn1(x))

        x = self.pool1(x)
        #print('after pool1', x.size())
        x = self.conv2(x)
        # print('after conv2', x.size())
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        # print('after pool2', x.size())
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        #print('after pool4', x.size())
        #x = F.avg_pool2d(x, x.shape[-1])
        # print('after pool2d', x.size())
        # x = x.permute(0, 2, 1)
        # print('after permute')
        x = self.flat(x)
        #print('after flatten', x.size())
        x = self.fc1(x)
        x = self.fc2(x)
        # print('after fc1', x.size())
        return F.log_softmax(x, dim=1)