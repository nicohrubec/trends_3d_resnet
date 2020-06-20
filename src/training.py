import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src import configs
from src.model import BaselineTab
from src.utils import weighted_nae


def train_fold(train_loader, val_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = weighted_nae
    model = BaselineTab().to(device)
    optimizer = optim.Adam(model.parameters(), lr=configs.lr, weight_decay=1e-5)

    # training
    for epoch in range(configs.epochs):
        print("Training epoch ", epoch+1)
        trn_loss = 0.0
        val_loss = 0.0

        for i, batch_data in enumerate(train_loader):
            tab, label = batch_data
            tab, label = tab.to(device), label.to(device)

            optimizer.zero_grad()
            logits = model(tab)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            trn_loss += loss.item() / len(train_loader)

        with torch.no_grad():
            print("Train loss: {:.4f}".format(trn_loss))


