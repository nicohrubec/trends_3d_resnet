from src import preprocess

compression = True

if __name__ == '__main__':
    if compression:
        preprocess.compress_images(mode='train')
        preprocess.compress_images(mode='test')
