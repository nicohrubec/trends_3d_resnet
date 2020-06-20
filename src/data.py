from src import configs

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold


def get_dataloader(mode='train', fold_index=0):
    if mode == 'train':
        train_set = TReNDsDataset(mode='train', fold_index=fold_index)
        train_loader = DataLoader(train_set, batch_size=configs.train_batch_size, shuffle=True,
                                  num_workers=configs.num_workers)
        val_set = TReNDsDataset(mode='val', fold_index=fold_index)
        val_loader = DataLoader(val_set, batch_size=configs.test_batch_size, shuffle=False,
                                num_workers=configs.num_workers)

        return train_loader, val_loader

    elif mode == 'test':
        test_set = TReNDsDataset(mode='test', fold_index=None)
        test_loader = DataLoader(test_set, batch_size=configs.test_batch_size, shuffle=False,
                                 num_workers=configs.num_workers)

        return test_loader


class TReNDsDataset(Dataset):

    def __init__(self, mode, fold_index):
        self.mode = mode
        self.fold_index = fold_index
        self.gkf = GroupKFold(n_splits=3)  # groupkfold on id --> same patient never in same fold
        self.all_samples = []

        if self.mode == 'train':
            self.augment = True
        else:
            self.augment = False

        train = pd.read_csv(configs.labels).sort_values(by='Id')
        train['isTrain'] = True
        loadings = pd.read_csv(configs.loadings_file)
        fncs = pd.read_csv(configs.fnc_matrix)
        schaefer_train = np.load(configs.train_schaefer_npy)
        schaefer_test = np.load(configs.test_schaefer_npy)
        targets = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']
        tabular_features = list(fncs.columns) + list(loadings.columns[1:])

        # merge loadings and fnc to lables
        data = pd.merge(loadings, train, on='Id', how='left')  # this will be merged on the flattened fncs
        ids_fnc = np.repeat(data.Id, 53)
        fncs['Id'] = ids_fnc.values
        data = pd.merge(fncs, data, on='Id', how='left')

        # split train test and drop unnecessary columns
        data['isTrain'].fillna(False, inplace=True)
        df = data[data.isTrain == True]
        df.drop('isTrain', axis=1, inplace=True)
        test_df = data[data.isTrain == False]
        test_df.drop(targets+['isTrain'], inplace=True, axis=1)
        df.dropna(inplace=True)

        if mode == 'train' or mode == 'val':
            # select training or validation set for current fold
            for fold_id, (train_idx, val_idx) in enumerate(self.gkf.split(df, groups=df.Id)):
                if fold_index == fold_id:
                    if mode == 'train':
                        idx = train_idx
                    elif mode == 'val':
                        idx = val_idx

            # select samples and split into subsets --> id, tabular, labels
            df = df.iloc[idx]
            df_tabular = np.asarray(df[tabular_features])
            df_tabular = np.hstack((df_tabular, schaefer_train[idx]))
            df_labels = np.asarray(df[targets])
            df_id = np.asarray(df['Id'])

            # add subsets to list of all samples to index from
            component_counter = 0
            for i in range(len(df)):
                filename = configs.data_dir / 'fMRI_train_npy' / (str(df_id[i])+'_'+str(component_counter)+'.npz')
                self.all_samples.append([df_id[i], filename, df_tabular[i], df_labels[i]])

                if component_counter < 52:
                    component_counter += 1
                else:
                    component_counter = 0

        elif mode == 'test':
            test_df_tabular = np.asarray(test_df[tabular_features])
            test_df_tabular = np.hstack((test_df_tabular, schaefer_test))
            test_df_id = np.asarray(test_df['Id'])

            component_counter = 0
            for i in range(len(test_df)):
                filename = configs.data_dir / 'fMRI_test_npy' / (str(test_df_id[i]) + '_' + str(component_counter) + '.npz')
                self.all_samples.append([test_df_id[i], filename, test_df_tabular[i]])

                if component_counter < 52:
                    component_counter += 1
                else:
                    component_counter = 0

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        if self.mode == 'train' or self.mode == 'val':
            id, filename, tabular, labels = self.all_samples[idx]
            # img = np.load(filename)['arr_0'].astype(np.float32)
            # img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))

            if self.augment:  # image augmentations to be implemented
                pass

            # return torch.FloatTensor(img), torch.FloatTensor(tabular), torch.FloatTensor(labels)
            return torch.FloatTensor(tabular), torch.FloatTensor(labels)

        elif self.mode == 'test':
            id, filename, tabular = self.all_samples[idx]
            # img = np.load(filename)['arr_0'].astype(np.float32)
            # img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))

            # return torch.FloatTensor(img), torch.FloatTensor(tabular)
            return torch.FloatTensor(tabular)
