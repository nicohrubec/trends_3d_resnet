from src import preprocess, utils, data, configs, training, eval, inference
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
            print("Training of fold {}".format(fold))
            train_loader, val_loader = data.get_dataloader(mode=configs.mode, fold_index=fold)
            training.train_fold(train_loader, val_loader, fold_index=fold)

    elif configs.mode == 'test':

        for fold in [0, 1, 2]:
            train_loader, val_loader = data.get_dataloader(mode='train', fold_index=fold)
            eval.eval_fold(val_loader, fold_index=fold)

        test_loader = data.get_dataloader(mode=configs.mode)
        preds = inference.predict_test(test_loader)
        utils.create_submission(preds)
