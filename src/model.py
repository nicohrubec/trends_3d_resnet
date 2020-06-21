import torch
import torch.nn as nn
import torch.nn.functional as F

from src import resnet

class Baseline(torch.nn.Module):

    def __init__(self):
        super(Baseline, self).__init__()
        self.img_conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.img_conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.img_out = nn.Linear(64, 128)

        self.tab_lin1 = nn.Linear(79, 128)
        self.tab_lin2 = nn.Linear(128, 128)

        self.out = nn.Linear(256, 5)

    def forward(self, img, tab):
        img = self.img_conv1(img)
        img = self.maxpool(img)
        img = F.relu(img)
        img = self.img_conv2(img)
        img = self.maxpool(img)
        img = F.relu(img)
        img = torch.squeeze(F.adaptive_avg_pool3d(img, (1, 1, 1)))
        img = self.img_out(img)

        tab = self.tab_lin1(tab)
        tab = F.relu(tab)
        tab = self.tab_lin2(tab)
        tab = F.relu(tab)

        out = torch.cat((img, tab), dim=1)
        out = self.out(out)

        return out


class BaselineTab(torch.nn.Module):

    def __init__(self):
        super(BaselineTab, self).__init__()
        self.img_conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.img_conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.img_out = nn.Linear(64, 128)
        self.tab_lin1 = nn.Linear(479, 1024)
        self.tab_lin2 = nn.Linear(1024, 1024)
        self.tab_lin3 = nn.Linear(1024, 1024)
        self.tab_lin4 = nn.Linear(1024, 1024)

        self.out = nn.Linear(1024+128, 5)

    def forward(self, img, tab):
        img = self.img_conv1(img)
        img = self.maxpool(img)
        img = F.relu(img)
        img = self.img_conv2(img)
        img = self.maxpool(img)
        img = F.relu(img)
        img = torch.squeeze(F.adaptive_avg_pool3d(img, (1, 1, 1)))
        img = self.img_out(img)

        tab = self.tab_lin1(tab)
        tab = F.relu(tab)
        tab = self.tab_lin2(tab)
        tab = F.relu(tab)
        tab = self.tab_lin3(tab)
        tab = F.relu(tab)
        tab = self.tab_lin4(tab)
        tab = F.relu(tab)

        out = torch.cat((img, tab), 1)
        out = self.out(out)

        return out


class BaselineResnet(torch.nn.Module):

    def __init__(self):
        super(BaselineResnet, self).__init__()
        self.resnet = resnet.resnet3d_10(shortcut_type='B')
        self.tab_lin1 = nn.Linear(479, 1024)
        self.tab_lin2 = nn.Linear(1024, 1024)
        self.tab_lin3 = nn.Linear(1024, 1024)
        self.tab_lin4 = nn.Linear(2048, 2048)

        self.out = nn.Linear(2048, 5)

    def forward(self, img, tab):
        img = self.resnet(img)

        tab = self.tab_lin1(tab)
        tab = F.relu(tab)
        tab = self.tab_lin2(tab)
        tab = F.relu(tab)
        tab = self.tab_lin3(tab)
        tab = F.relu(tab)
        x = torch.cat((tab, img), 1)
        x = self.tab_lin4(x)
        x = F.relu(x)
        out = self.out(x)

        return out
