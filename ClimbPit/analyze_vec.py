import ChineseList
from ClimbPit import before_reduction
import argparse
import sys
import struct
import numpy as np


def read_vec(fname, method):
	vocab = {}

	if method == 'svd':
		decoder = struct.Struct('1000f')
		mat = []
		EOF = False
		num_lines = 0
		
		with open(fname, 'rb') as fr:
			while True:
				chars = []
				while True:
					ch = fr.read(1)
					if not ch:
						EOF = True
						break
					if ch == b' ':
						break
					chars.append(ch)
					if len(chars) > 100:
						raise ValueError

				if EOF:
					break
				word = b''.join(chars).decode()
				vec = fr.read(decoder.size)
				fr.read(1) # trailing new line

				mat.append(np.array( decoder.unpack(vec) ).reshape(1, -1))
				vocab[word] = len(vocab)

				num_lines += 1
				if num_lines % 10000 == 0:
					print('\rHave read {:d}K words'.format(num_lines//1000), end='')

		mat = np.array(mat)
		print('\nThe array shape is {}'.format(mat.shape))

		return mat, vocab

	elif method == 's2v':
		with open(fname, 'r', encoding='utf8') as fr:
			for i, line in enumerate(fr):
				if i == 0:
					V, D = map(int, line.split()[:2])

				elif i <= V:
					word, _ = line.split(maxsplit=1)
					vocab[word] = len(vocab)
					if i % 10000 == 0:
						print('\rHave read {:d}K words'.format(i//1000), end='')

				elif i <= V + 1000:
					continue

				else:
					print('\nBuilding matrix now...')
					vec = line.split()
					assert len(vec) == V * D
					mat = [vec[j*D : (j+1)*D] for j in range(V)]

		mat = np.array(mat).reshape(V, 1, -1)
		assert mat.shape == (V, 1, D)
		return mat, vocab

	elif method == 's2v_sememe':
		with open(fname, 'r', encoding='utf8') as fr:
			mat = []
			
			for i, line in enumerate(fr):
				if i == 0:
					V, D = map(int, line.split()[:2])

				elif i <= V:
					parts = line.split()
					word = parts[0]
					vec = list( map(float, parts[1:]) )
					vocab[word] = len(vocab)
					mat.append(vec)

		mat = np.array(mat).reshape(V, 1, -1)
		assert mat.shape == (V, 1, D)
		return mat, vocab

	else:
		raise NotImplementedError



def parse_args(argv):
	parser = argparse.ArgumentParser(
		description='Transform binary vectors to text file.')

	parser.add_argument('input', type=str)
	parser.add_argument('method', type=str)
	args = parser.parse_args(argv)
	return args


def main(args):
	mat, vocab = read_vec(args.input, args.method)
	before_reduction.main(mat, vocab)


if __name__ == '__main__':
	args = parse_args(sys.argv[1:])
	main(args)
