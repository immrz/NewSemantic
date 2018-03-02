import os

RAW_LIST = r'E:\users\leifa\data\ch.list.count.txt'
DATA_PATH = r'E:\users\v-rumao\datasets\chinese_lists'

_version = 4
_filtered = 'filtered_{:02d}.txt'.format(_version)
_vocab = 'vocab_{:02d}.txt'.format(_version)
_vector = 'truncated_vectors_{:02d}.npy'.format(_version)

LIST_FILE = os.path.join(DATA_PATH, _filtered)
LIST_VOCAB = os.path.join(DATA_PATH, _vocab)
VECTORS = os.path.join(DATA_PATH, _vector)
