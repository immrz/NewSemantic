import ChineseList
import os
from reduce_dim import read_vocab


def extract_id(vocab, fname):
    word_to_id = {}
    with open(fname, 'r', encoding='utf8') as fr:
        for line in fr:
            ix, word = line.strip().split(maxsplit=1)
            if word in vocab:
                word_to_id[word] = ix
    return word_to_id


def main():
    graph_dir = r'E:\users\v-rumao\datasets\zhGraph'
    dict_file = 'step4_Id2Word.txt'
    graph_file = 'step4_SimpleGraph.txt'

    list_vocab = read_vocab(ChineseList.LIST_VOCAB)
    word_to_id = extract_id(list_vocab, os.path.join(graph_dir, dict_file))
    with open(os.path.join(graph_dir, 'word_to_id.txt'), 'w', encoding='utf8') as fw:
        for w, v in word_to_id.items():
            fw.write('{} {}\n'.format(w, v))


if __name__ == '__main__':
    main()