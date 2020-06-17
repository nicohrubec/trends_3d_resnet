from src import preprocess, utils

# hyperparameters
compression = True
SEED = 42

if __name__ == '__main__':
    if compression:
        preprocess.compress_images(mode='train')
        preprocess.compress_images(mode='test')

    utils.set_seed(SEED)
    
