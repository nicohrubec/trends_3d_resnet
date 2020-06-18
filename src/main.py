from src import preprocess, utils, data

# hyperparameters
compression = False
calc_fnc_matrix = True
SEED = 0
mode = 'train'  # train or test phase
folds = [0]  # list of folds to train on --> provide n indices from 0 to n_splits-1

if __name__ == '__main__':
    if compression:
        preprocess.compress_images(mode=mode)
    if calc_fnc_matrix:
        preprocess.calc_fnc_matrix()  # extract ordered fnc matrix for later merging here

    utils.set_seed(SEED)
    
