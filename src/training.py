import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from apex import amp

from src import configs
from src.model import BaselineTab
from src.utils import weighted_nae, weighted_nae_npy


def train_fold(train_loader, val_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = weighted_nae
    model = BaselineTab().to(device)
    optimizer = optim.Adam(model.parameters(), lr=configs.lr, weight_decay=1e-5)

    if configs.use_amp:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2',
            loss_scale='dynamic'
        )

    # training
    for epoch in range(configs.epochs):
        print("Training epoch ", epoch+1)
        trn_loss = 0.0
        val_loss = 0.0

        oof = []
        oof_targets = []

        for i, batch_data in enumerate(train_loader):
            img, tab, label = batch_data
            img, tab, label = img.to(device), tab.to(device), label.to(device)

            model.train()
            optimizer.zero_grad()
            logits = model(tab)
            loss = criterion(logits, label)
            
            if configs.use_amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()

            trn_loss += loss.item() / len(train_loader)

        with torch.no_grad():
            for i_val, val_batch_data in enumerate(val_loader):
                img, tab, label = val_batch_data
                img, tab, label = img.to(device), tab.to(device), label.to(device)

                model.eval()
                val_logits = model(tab)
                valid_loss = criterion(val_logits, label)

                val_loss += valid_loss.item() / len(val_loader)

                oof.append(val_logits.data.cpu().numpy())
                oof_targets.append(label.data.cpu().numpy())

            # prepare mean over components
            oof = np.concatenate(oof, axis=0)
            oof = oof.reshape((-1, 53, 5))
            oof = np.mean(oof, axis=1)
            oof_targets = np.concatenate(oof_targets, axis=0)
            oof_targets = oof_targets.reshape((-1, 53, 5))
            oof_targets = oof_targets[:, 0, :]

            # validate with mean over components
            score, domain_losses = weighted_nae_npy(oof_targets, oof)

            print("Train loss: {:.4f}".format(trn_loss))
            print("Val loss: {:.4f}".format(val_loss))
            print("Mean val loss: {:.4f}".format(score))
            print(domain_losses)


