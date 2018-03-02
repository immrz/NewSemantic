import ChineseList
from collections import defaultdict
import os
import struct


def write_binary(trans, fname):
    cnt = 0
    with open(fname, 'wb') as fw:
        for k, v in trans.items():
            l = len(v)
            pre = '{} {} '.format(k, l)
            fmt = '{}i'.format(l)
            contain = struct.pack(fmt, *v)
            fw.write(bytes(pre, encoding='utf8') + contain + b'\n')

            cnt += 1
            if cnt % 10000 == 0:
                print('\rHave written {:d}K words.'.format(cnt // 1000), end='')
    print('\nsave end')


def main():
    trans = defaultdict(list)
    with open(ChineseList.LIST_FILE, 'r', encoding='utf8') as fr:
        for i, line in enumerate(fr):
            words = line.split()[:-1]
            for word in words:
                trans[word].append(i)
            if i % 10000 == 0:
                print('\rHave read {:d}K lists.'.format(i // 1000), end='')
    print(i)

    assert len(trans) == 354725
    print('\nbegin writing...')
    # write_binary(trans, os.path.join(ChineseList.DATA_PATH, 'transformed_04.bin'))
    return trans


if __name__ == '__main__':
    trans = main()