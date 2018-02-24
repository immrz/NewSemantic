import ChineseList
import time
import numpy as np


def read_vocab(fname):
    vocab = []
    with open(fname, 'r', encoding='utf8') as fr:
        for line in fr:
            vocab.append(line.split()[0])
    return vocab


def write_vec(vec, vocab, fname):
    with open(fname, 'w', encoding='utf8') as fw:
        for w, v in zip(vocab, vec.tolist()):
            temp = ' '.join(map(str, v))
            fw.write('{} {}\n'.format(w, temp))


def main():
    vocab_f = ChineseList.LIST_VOCAB
    vec_f = ChineseList.VECTORS
    t_start = time.time()

    print('vocab file: {}\nvector file: {}'.format(vocab_f, vec_f))
    vocab = read_vocab(vocab_f)
    t_vocab = time.time()
    print('reading vocab takes {:.3f}s.'.format(t_vocab - t_start))

    vec = np.load(vec_f, allow_pickle=False)
    t_vec = time.time()
    print('loading vectors takes {:.3f}s.'.format(t_vec - t_vocab))

    assert vec.shape[0] == len(vocab), 'row number not correct'
    assert vec.shape[1] == 1000, 'column number not correct'

    write_vec(vec, vocab, r'E:\users\v-rumao\codes\Sememe\word_sememe.txt')
    t_end = time.time()
    print('writing vectors takes {:.3f}s.'.format(t_end - t_vec))


if __name__ == '__main__':
    main()