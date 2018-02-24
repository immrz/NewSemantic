from collections import defaultdict
import ChineseList


def compare_vocab(list_vocab, train_vocab_file):
    """
    Compare the vocabulary of the lists with that of the training file.

    :param list_vocab: Should be a set instead of a dict.
    :param train_vocab_file: str
    :return:
    """

    train_vocab = set()
    with open(train_vocab_file, 'r', encoding='utf8') as fr:
        for line in fr:
            word, _ = line.split()
            train_vocab.add(word)

    print('List: {:,d}\nTrain: {:,d}'.format(len(list_vocab), len(train_vocab)))
    print('List - Train: {:,d}'.format(len(list_vocab - train_vocab)))
    print('Train - List: {:,d}'.format(len(train_vocab - list_vocab)))


def write_vocab(fname, vocab):
    pairs = vocab.items()
    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

    with open(fname, 'w', encoding='utf8') as fw:
        for word, cnt in pairs:
            fw.write('{} {}\n'.format(word, cnt))


def main():
    vocab = defaultdict(lambda: 0)
    num_tokens = 0

    with open(ChineseList.LIST_FILE, 'r', encoding='utf8') as fr:
        for line in fr:
            words = line.split()[:-1]
            num_tokens += len(words)

            for w in words:
                vocab[w] += 1

    write_vocab(ChineseList.LIST_VOCAB, vocab)
    compare_vocab(set(vocab.keys()), r'E:\users\v-rumao\codes\Sememe\SE-WRL-master\datasets\VocabFile')

    print('There are totally {:,d} words and {:,d} tokens'
          ' in the list.'.format(len(vocab), num_tokens))


if __name__ == '__main__':
    main()
