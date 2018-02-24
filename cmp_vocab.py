import ChineseList
import os


def main():
    # Read list vocab
    list_vocab = set()
    with open(ChineseList.LIST_VOCAB, 'r', encoding='utf8') as fr:
        for line in fr:
            list_vocab.add(line.split()[0])

    # Read training vocab
    train_vocab = []
    vocab_file = r'E:\users\v-rumao\codes\Sememe\SE-WRL-master\datasets\VocabFile'
    total_num = 0

    with open(vocab_file, 'r', encoding='utf8') as fr:
        for line in fr:
            word, cnt = line.split()
            cnt = int(cnt)
            total_num += cnt
            train_vocab.append((word, cnt))

    # compare
    not_in_list_num = 0
    not_in_list = []
    for word, cnt in train_vocab:
        if word in list_vocab:
            continue
        not_in_list_num += cnt
        not_in_list.append('{} {}\n'.format(word, cnt))

    with open(os.path.join(ChineseList.DATA_PATH, 'not_in_list_vocab.txt'), 'w', encoding='utf8') as fw:
        fw.writelines(not_in_list)

    print('total: {}\nnot in list: {}'.format(total_num, not_in_list_num))
    print('ratio: {:.5f}'.format(not_in_list_num / total_num))


if __name__ == '__main__':
    main()