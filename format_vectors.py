import ChineseList
import time
import numpy as np
import struct
import os


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


def write_binary(mat, vocab, fname):
    encoder = struct.Struct('1000f')
    mat_list = mat.tolist()

    with open(fname, 'wb') as fw:
        for i in range(len(vocab)):
            word = bytes(vocab[i], encoding='utf8')
            vec = encoder.pack(*mat_list[i])
            fw.write(word + b' ' + vec + b'\n')


def main():
    vocab_f = ChineseList.LIST_VOCAB
    # vec_f = ChineseList.VECTORS
    vec_f = os.path.join(ChineseList.DATA_PATH, 'nmf_result_iter100.npy')
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

    # write_vec(vec, vocab, r'E:\users\v-rumao\codes\Sememe\word_sememe.txt')
    write_binary(vec, vocab, os.path.join(ChineseList.DATA_PATH, 'nmf_sememe_iter100.bin'))
    t_end = time.time()
    print('writing vectors takes {:.3f}s.'.format(t_end - t_vec))


if __name__ == '__main__':
    main()