import os
import multiprocessing as mp
import ChineseList
from ChineseList import count_words


def process_one_line(line):
    """
    :return:
        A tuple like ([],int,int) where the first element is
        the list of words, and the second is the tf.
    """

    l, df, tf = line.split()
    l = l.strip('[]\" \t\r\n')
    l = l.split('\",\"')
    return l, int(df), int(tf)


def criterion(record):
    """
    If some requirements are satisfied, delete this list.
    :param record:
    :return:
    """

    words, df, tf = record
    if df < 5 and len(words) < 20:
        return True
    if len(words) < 7:
        return True
    return False


def chunkify(fname, size=1024*1024):
    file_end = os.path.getsize(fname)

    with open(fname, 'rb') as fr:
        chunk_end = fr.tell()

        while True:
            chunk_start = chunk_end

            fr.seek(size, 1)
            fr.readline()

            chunk_end = fr.tell()
            yield chunk_start, chunk_end - chunk_start

            if chunk_end > file_end:
                break


def producer(fname, queue, chunk_start, chunk_size, to_del=criterion):
    """
    Read the raw list file and put the lists which need not be
    deleted into the queue to be written.
    :return:
    """

    with open(fname, 'rb') as fr:
        fr.seek(chunk_start)
        lines = fr.read(chunk_size).splitlines()
        num = len(lines)
        for line in lines:
            processed = process_one_line(line.decode())
            if to_del(processed):
                continue
            queue.put(' '.join(processed[0])
                      + ' {}\n'.format(processed[1]))
    return num


def consumer(fname, queue):
    """
    Accept the filtered lists and output them.
    :return:
    """

    num = 0
    with open(fname, 'w', encoding='utf8') as fw:
        buffer = []

        while True:
            msg = queue.get()

            if msg == '--kill--':
                break

            if len(buffer) < 200:
                buffer.append(str(msg))
                num += 1

            else:
                fw.write(''.join(buffer))
                buffer = []

        if buffer:
            fw.write(''.join(buffer))

    return num


def main(flt_list=ChineseList.LIST_FILE, to_del=criterion):
    raw_list = ChineseList.RAW_LIST
    # flt_list = ChineseList.LIST_FILE

    print('raw list: {}'.format(raw_list))
    print('filtered list: {}'.format(flt_list))

    pool = mp.Pool(40)
    manager = mp.Manager()
    queue = manager.Queue()

    listener = pool.apply_async(consumer, (flt_list, queue))
    workers = []

    for chunk_start, chunk_size in chunkify(raw_list):
        args = (raw_list, queue, chunk_start, chunk_size)
        kwds = {'to_del':to_del}
        workers.append(pool.apply_async(producer, args, kwds))

    before = [worker.get() for worker in workers]
    queue.put('--kill--')
    after = listener.get()

    pool.close()
    pool.join()

    print('Before filtering, there are {:,d} lists.'.format(sum(before)))
    print('After filtering, there are {:,d} lists.'.format(after))

    # count_words.main()


if __name__ == '__main__':
    main()
