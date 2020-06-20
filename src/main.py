from src import preprocess, utils, data, configs, training
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    if configs.compression:
        preprocess.compress_images(mode=configs.mode)
    if configs.calc_fnc_matrix:
        preprocess.calc_fnc_matrix()  # extract ordered fnc matrix for later merging here
    if configs.preprocess_schaefer:
        preprocess.transform_schaefer()

    utils.set_seed(configs.SEED)

    if configs.mode == 'train':
        for fold in configs.folds:
            train_loader, val_loader = data.get_dataloader(mode=configs.mode, fold_index=fold)
            training.train_fold(train_loader, val_loader)

    elif configs.mode == 'test':
        test_loader = data.get_dataloader(mode=configs.mode)


