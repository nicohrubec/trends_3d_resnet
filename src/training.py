import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src import configs
from src.model import Baseline


def train_fold(train_loader, val_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()  # TODO: Implement competition metric.
    model = Baseline().to(device)
    optimizer = optim.Adam(model.parameters(), lr=configs.lr, weight_decay=1e-5)

    # training
    for epoch in range(configs.epochs):
        print("Training epoch ", epoch+1)
        trn_loss = 0.0
        val_loss = 0.0

        for i, batch_data in enumerate(train_loader):
            if i > 0:
                break
            img, tab, label = batch_data
            img, tab, label = img.to(device), tab.to(device), label.to(device)

            optimizer.zero_grad()
            logits = model(img, tab)
            loss = criterion(logits[label==label], label[label==label])
            loss.backward()
            optimizer.step()
            print(loss)


