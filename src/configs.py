from pathlib import Path

project_dir = Path.cwd().parent
data_src = Path("D:/trends_files/")
data_dir = project_dir / 'data'
model_dir = project_dir / 'models'
sub_dir = project_dir / 'subs'

fnc_file = data_dir / 'fnc.csv'
fnc_match_file = data_dir / 'ICN_numbers.csv'
loadings_file = data_dir / 'loading.csv'
site2 = data_dir / 'reveal_ID_site2.csv'
labels = data_dir / 'train_scores.csv'
sample_sub = data_dir / 'sample_submission.csv'
fnc_matrix = data_dir / 'fnc_matrix.csv'
train_schaefer = data_dir / 'training_data_schaefer18_400.csv'
test_schaefer = data_dir / 'test_data_schaefer18_400.csv'
train_schaefer_npy = data_dir / 'train_rois.npy'
test_schaefer_npy = data_dir / 'test_rois.npy'


# hyperparameters
compression = False
calc_fnc_matrix = False
preprocess_schaefer = True
SEED = 0
mode = 'train'  # train or test phase
folds = [0]  # list of folds to train on --> provide n indices from 0 to n_splits-1
train_batch_size = 64
test_batch_size = train_batch_size * 4
num_workers = 1
lr = .0003
epochs = 10
