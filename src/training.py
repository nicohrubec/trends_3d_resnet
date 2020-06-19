import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src import configs
from src.model import BaselineTab


def train_fold(train_loader, val_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()  # TODO: Implement competition metric.
    model = BaselineTab().to(device)
    optimizer = optim.Adam(model.parameters(), lr=configs.lr, weight_decay=1e-5)

    # training
    for epoch in range(configs.epochs):
        print("Training epoch ", epoch+1)
        trn_loss = 0.0
        val_loss = 0.0

        for i, batch_data in enumerate(train_loader):
            print(i/len(train_loader))
            tab, label = batch_data
            tab, label = tab.to(device), label.to(device)

            optimizer.zero_grad()
            logits = model(tab)
            loss = criterion(logits[label==label], label[label==label])
            loss.backward()
            optimizer.step()
            print(loss)


