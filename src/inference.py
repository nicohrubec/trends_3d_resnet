import os
import torch
import numpy as np

from src import configs
from src.model import BaselineResnet


def predict_test(test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BaselineResnet().to(device)
    preds = []

    for fold_folder in os.listdir(configs.submission_folder):
        fold_dir = configs.submission_folder / fold_folder

        for i, model_checkpoint in enumerate(os.listdir(fold_dir)):
            model_checkpoint_path = fold_dir / model_checkpoint
            model.load_state_dict(torch.load(model_checkpoint_path), strict=False)
            model_preds = []

            with torch.no_grad():
                for i_test, test_batch_data in enumerate(test_loader):
                    img, tab = test_batch_data
                    img, tab = img.to(device), tab.to(device)

                    model.eval()
                    test_logits = model(img, tab)

                    model_preds.append(test_logits.data.cpu().numpy())

                model_preds = np.concatenate(model_preds, axis=0)
                model_preds = model_preds.reshape((-1, 53, 5))
                model_preds = np.mean(model_preds, axis=1)

            preds.append(model_preds)

    preds = np.concatenate(preds, axis=1)
    preds = np.reshape(preds, (len(preds), -1, 5))
    preds = np.mean(preds, axis=1)

    return preds