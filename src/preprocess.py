from src import configs

import os
import numpy as np
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
