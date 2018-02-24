from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
import ChineseList
import time
from sklearn.decomposition import TruncatedSVD, NMF
import sys
import traceback
import numpy as np
import os


def read_vocab(fname):
    """
    Read the vocabulary from the given file so that the orders
    of the rows of the sparse matrix are controlled.

    :param fname:
    :return:
        vocab: A dictionary mapping from words to indices.
    """

    vocab = {}

    with open(fname, 'r', encoding='utf8') as fr:
        for line in fr:
            w, _ = line.split()
            assert w not in vocab, 'Word {} already exists!'.format(w)

            vocab[w] = len(vocab)

    return vocab


def read_sparse_matrix(list_file, vocab_file, use_tfidf=False):
    """
    Read a sparse term-document matrix from the given file.

    :return:
        A sparse matrix of type `scipy.sparse.csc_matrix` where
            each row represents a term and each column represents
            a document.
    """

    data = []
    indices = []
    indptr = [0]
    vocab = read_vocab(vocab_file)
    N = 5814998

    word_df = [0] * len(vocab)

    print('Reading {}.'.format(list_file))

    with open(list_file, 'r', encoding='utf8') as fr:
        # each line in the file corresponds
        # to a column in the sparse matrix.
        for i, line in enumerate(fr):
            if i % 10000 == 0:
                print('\rProgress: {:.3f}%'.format(i/N*100), end='')

            parts = line.split()
            words, tf = parts[:-1], int(parts[-1])

            for w in words:
                indices.append(vocab[w])
                data.append(tf)
                word_df[vocab[w]] += 1

            indptr.append(len(indices))

    if use_tfidf:
        nu = len(indptr) - 1
        for i in range(len(data)):
            data[i] *= np.log(nu / (1 + word_df[indices[i]]))


    print('\nlength of data: {:,d}.\nlength of indices: {:,d}.\n'
          'length of indptr: {:,d}.\n'.format(len(data), len(indices), len(indptr)))
    return csc_matrix((data, indices, indptr), dtype=np.float32).tocsr(), vocab


def apply_pca(mat, num_col, save_file):
    """Try to apply pca on the given matrix and keep certain components."""
    print('Now applying pca on the sparse matrix.')

    # sklearn doesn't work!!!
    svd = TruncatedSVD(n_components=num_col, n_iter=10)
    try:
        truncated = svd.fit_transform(mat)
        np.save(save_file, truncated, allow_pickle=False)
        return truncated
    except Exception:
        print('Reduction failed!')
        traceback.print_exc(file=sys.stdout)
        return mat

    # scipy svd too slow!
    # try:
    #     u, s, _ = svds(mat, k=num_col, maxiter=5, return_singular_vectors='u')
    #     print('The shapes of u and s are {} and {}'
    #           ' respectively.'.format(u.shape, s.shape))
    #     return u.dot(np.diag(s))
    # except Exception:
    #     print('Reduction failed!')
    #     traceback.print_exc(file=sys.stdout)
    #     return mat


def apply_nmf(mat, num_col, save_file):
    print('Now applying nmf on the sparse matrix.')
    nmf = NMF(n_components=num_col, init='nndsvda', solver='mu', random_state=59, alpha=0.1, max_iter=20, verbose=True)

    try:
        truncated = nmf.fit_transform(mat)
        np.save(save_file, truncated, allow_pickle=False)
        print('real iterations: {}\n'
              'final error'.format(nmf.n_iter_, nmf.reconstruction_err_))
        return truncated
    except Exception:
        print('Reduction failed!')
        traceback.print_exc(file=sys.stdout)
        return mat


def main():
    t_start = time.time()

    vocab_size = 354725
    list_size = 5814998
    num_col = 1000

    list_file = ChineseList.LIST_FILE
    vocab_file = ChineseList.LIST_VOCAB

    mat, _ = read_sparse_matrix(list_file, vocab_file)
    assert mat.shape == (vocab_size, list_size), 'matrix shape not matched'
    t_mat = time.time()
    print('Reading lists and building the sparse matrix'
          ' take totally {:.3f}s.'.format(t_mat - t_start))

    # trunc_mat = apply_pca(mat, num_col, ChineseList.VECTORS)
    trunc_mat = apply_nmf(mat, num_col, os.path.join(ChineseList.DATA_PATH, 'nmf_result.npy'))
    t_pca = time.time()
    print('The dimensionality reduction takes {:.3f}s.'.format(t_pca - t_mat))
    print('Before reduction, the shape is {}\n'
          'After reduction, the shape is {}'
          ' and the type is {}.'.format(mat.shape, trunc_mat.shape, type(trunc_mat)))


if __name__ == '__main__':
    main()
