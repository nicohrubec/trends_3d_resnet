import os
import torch
import numpy as np

from src import configs
from src.model import BaselineTab
from src.utils import weighted_nae_npy


def eval_fold(val_loader, fold_index):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BaselineTab().to(device)

    fold_dir = configs.submission_folder / ('fold_{}'.format(fold_index))
    oof_sum = []

    if os.path.exists(fold_dir):  # check if directory is properly setup
        for model_checkpoint in os.listdir(fold_dir):
            model_checkpoint_path = fold_dir / model_checkpoint
            model.load_state_dict(torch.load(model_checkpoint_path), strict=False)

            oof = []
            oof_targets = []

            with torch.no_grad():
                for i_val, val_batch_data in enumerate(val_loader):
                    img, tab, label = val_batch_data
                    img, tab, label = img.to(device), tab.to(device), label.to(device)

                    model.eval()
                    val_logits = model(img, tab)

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
                print("Eval fold {}:".format(fold_index))
                print("Mean val loss: {:.4f}".format(score))
                print(domain_losses)

            oof_sum.append(oof)

        # stack all model preds for fold
        oof_sum = np.concatenate(oof_sum, axis=1)
        oof_sum = np.reshape(oof_sum, (len(oof_sum), -1, 5))
        oof_sum = np.mean(oof_sum, axis=1)  # ensemble all models with equal contribution

        score, domain_losses = weighted_nae_npy(oof_targets, oof_sum)
        print("Eval fold {}:".format(fold_index))
        print("Mean val loss: {:.4f}".format(score))
        print(domain_losses)
