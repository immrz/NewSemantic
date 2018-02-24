from functools import partial
import os
import ChineseList
from ChineseList import filter_lists


def not_in_my_vocab(vocab, record):
    words, _, _ = record
    if all([w not in vocab for w in words]):
        return True
    return False


def read_extra_vocab(files):

    def read_file(fname):
        s = set()
        name = fname.split(os.sep)[-1].split('.')[0]
        with open(fname, 'r', encoding='utf8') as fr:
            for line in fr:
                parts = line.split()
                if len(parts) < 3:
                    continue
                elif name == 'analogy':
                    s = s.union(parts)
                elif name.split('-')[0] == 'wordsim':
                    s = s.union(parts[:-1])
                else:
                    raise Exception
        return s

    vocab = set()
    for f in files:
        s = read_file(f)
        print('vocab of {} has size {}.'.format(f, len(s)))
        vocab = vocab.union(s)

    return vocab


def main():
    test_path = r'E:\users\v-rumao\codes\Sememe\SE-WRL-master\datasets'
    test_file = ['analogy.txt', 'wordsim-240.txt', 'wordsim-297.txt']

    join = partial(os.path.join, test_path)
    files = list(map(join, test_file))

    assert all([os.path.isfile(x) for x in files])
    vocab = read_extra_vocab(files)

    print('Extra vocab size is {:,d}'.format(len(vocab)))

    with open(r'E:\users\v-rumao\codes\Sememe\SE-WRL-master\datasets\eval_vocab.txt', 'w', encoding='utf8') as fw:
    	for w in vocab:
    		fw.write(w + '\n')

    if input('whether to print the vocab [p/n]: ') == 'p':
        print(vocab)
    if input('whether to continue [y/n]: ') != 'y':
        return

    to_del = partial(not_in_my_vocab, vocab)
    filter_lists.main(
        flt_list=os.path.join(ChineseList.DATA_PATH,
                              'extra_lists.txt'),
        to_del=to_del
    )


if __name__ == '__main__':
    main()