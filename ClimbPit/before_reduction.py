import ChineseList
from reduce_dim import read_sparse_matrix
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from functools import partial
import os
import multiprocessing as mp
from collections import defaultdict


def similarity(mat, vocab, fname):
    rank = []

    with open(fname, 'r', encoding='utf8') as fr:
        for i, line in enumerate(fr):
            w1, w2, _ = line.split()
            try:
                ix1, ix2 = vocab[w1], vocab[w2]
                sim = cosine_similarity(mat[ix1], mat[ix2])
                assert sim.shape == (1, 1)
                rank.append((i, sim[0,0]))
            except KeyError:
                print('{}-th line not in list vocab'.format(i + 1))

        rank_sorted = sorted(rank, key=lambda x: x[1], reverse=True)
        my_rank = [x[0] for x in rank_sorted]
        corr, _ = spearmanr(my_rank, sorted(my_rank))

    print('The correlation is {:.3f}\n'.format(corr))


def worker(mat, record):
    vec = [mat[i] for i in record[0]]
    v1 = vec[1] - vec[0] + vec[2]
    v2 = vec[3]

    v2_rank = 1
    v2_score = cosine_similarity(v1, v2)[0,0]

    for i in range(mat.shape[0]):
        if i in record[0]:
            continue
        score = cosine_similarity(v1, mat[i])[0,0]
        if score > v2_score:
            v2_rank += 1

    return v2_rank, record[1]


def analogy(mat, vocab, fname):
    records = []
    cur_ctg = None
    with open(fname, 'r', encoding='utf8') as fr:
        for i, line in enumerate(fr):
            if line[0] == ':':
                cur_ctg = line.split()[-1].split('-')[0]
                continue
            parts = line.split()
            assert len(parts) == 4

            try:
                to_ix = [vocab[x] for x in parts]
                records.append((to_ix, cur_ctg))
            except KeyError:
                print('{}-th line not in list vocab'.format(i + 1))

    print('Totally {} jobs to do'.format(len(records)))

    pool = mp.Pool(20)
    jobs = [pool.apply_async(worker, args=(mat, x)) for x in records]

    results = [j.get() for j in jobs]
    pool.close()
    pool.join()

    # results = []
    # for i, record in enumerate(records):
    #     results.append(worker(mat, record))
    #     print('\rProgress: {:.3f}'.format(i / len(records)), end='')
    # print('')

    eval_res = defaultdict(lambda: [0,0,0])
    for res in results:
        rank, ctg = res
        if rank == 1:
            eval_res[ctg][0] += 1
            eval_res['all'][0] += 1
        eval_res[ctg][1] += rank
        eval_res['all'][1] += rank
        eval_res[ctg][2] += 1
        eval_res['all'][2] += 1

    for k, v in eval_res:
        print('result for {}:'.format(k))
        print('Accuracy: {:.3f}'
              ' Mean Rank: {:.3f}'.format(v[0] / v[2], v[1] / v[2]))


def evaluate(mat, vocab, task):
    print('doing the task {} now'.format(task))

    task_type = os.path.basename(task)[:7]
    if task_type == 'wordsim':
        similarity(mat, vocab, task)
    elif task_type == 'analogy':
        analogy(mat, vocab, task)
    else:
        raise Exception('Undefined task here')


def main(mat=None, vocab=None):
    eval_dir = r'E:\users\v-rumao\codes\Sememe\SE-WRL-master\datasets'
    eval_task = ['wordsim-240.txt', 'wordsim-297.txt']

    eval_file = list(map(
        partial(os.path.join, eval_dir),
        eval_task
    ))

    for task in eval_file:
        print(task)

    if mat is None or vocab is None:
        # this matrix should be a sparse row-based one
        mat, vocab = read_sparse_matrix(
            ChineseList.LIST_FILE,
            ChineseList.LIST_VOCAB,
            onehot=True
        )

    for task in eval_file:
        evaluate(mat, vocab, task)


if __name__ == '__main__':
    main()