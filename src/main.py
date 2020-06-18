from src import preprocess, utils, data, configs

if __name__ == '__main__':
    if configs.compression:
        preprocess.compress_images(mode=configs.mode)
    if configs.calc_fnc_matrix:
        preprocess.calc_fnc_matrix()  # extract ordered fnc matrix for later merging here

    utils.set_seed(SEED)
    
