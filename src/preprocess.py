from src import configs

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


def compress_images(mode='train'):
    if mode == 'train':
        src_folder = configs.data_src / 'fMRI_train_npy'
        tar_folder = configs.data_dir / 'fMRI_train_npy'
    elif mode == 'test':
        src_folder = configs.data_src / 'fMRI_test_npy'
        tar_folder = configs.data_dir / 'fMRI_test_npy'

    if not os.path.exists(tar_folder):
        os.mkdir(tar_folder)

    scaler = StandardScaler()
    for file in tqdm(os.listdir(src_folder)):
        arr = np.load(src_folder / file)

        for component_id in range(53): # there is 53 subcomponents
            component = arr[:, :, :, component_id]
            component = component[2:49, 3:61, 1:50]  # strip manually observed all zero dimensions
            component = scaler.fit_transform(component.reshape(-1, 1))
            component[(component < 1) & (component > -1)] = 0.  # abs voxel value < 1 set 0
            component = np.reshape(component, (47, 58, 49))

            file_path = tar_folder / (file[:-4]+'_'+str(component_id)+'.npz')
            np.savez_compressed(file_path, component)


def calc_fnc_matrix():
    fncs = pd.read_csv(configs.fnc_file)
    ICN = pd.read_csv(configs.fnc_match_file)
    fncs.drop('Id', axis=1, inplace=True)

    net_set = ICN.ICN_number.to_list()
    # net_set = [int(net.split("(")[1].split(")")[0]) for net in net_set]  # extract subcomponent number

    features = np.zeros((len(fncs), len(net_set), len(net_set)))  # shape: samples x num_sub_networks x num_sub_networks

    # create matrix containing correlation features for each sample
    for col in fncs.columns:
        split = col.split('_')
        net1 = int(split[0].split("(")[1].split(")")[0])
        net2 = int(split[2].split("(")[1].split(")")[0])

        temp = fncs[[col]].values.reshape(-1)
        features[:, net_set.index(net1), net_set.index(net2)] = temp
        features[:, net_set.index(net2), net_set.index(net1)] = temp

    # create dataframe with one component = one sample
    features = features.reshape(-1, 53)
    features = pd.DataFrame(data=features, columns=net_set)
    path = configs.data_dir / 'fnc_matrix.csv'
    features.to_csv(path, index=False)
