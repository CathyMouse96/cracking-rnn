import argparse
import os
import numpy as np
import collections
from six.moves import cPickle as pickle

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default='data/rockyou/input.txt', help='input file in UTF-8')
parser.add_argument('--output_dir', type=str, default='preprocessed', help='output directory for preprocessed files')
parser.add_argument('--valid_frac', type=float, default='0.1', help='fraction of data to use as validation set (default: 0.1)')
parser.add_argument('--test_frac', type=float, default='0.2', help='fraction of data to use as test set (default: 0.2)')
parser.add_argument('--max_length', type=int, default='12', help='max length of passwords (default: 12)')
parser.add_argument('--verbose', action='store_true', help='verbose printing')
args = parser.parse_args()

if not os.path.exists(args.output_dir):
	os.makedirs(args.output_dir)

vocab_file = os.path.join(args.output_dir, 'vocab.pkl')
train_file = os.path.join(args.output_dir, 'train.txt')
valid_file = os.path.join(args.output_dir, 'valid.txt')
test_file = os.path.join(args.output_dir, 'test.txt')

lines = []

# skip passwords longer than 12 chars
with open(args.input_file, 'r') as f:
	for line in f:
		if len(line) > args.max_length + 1: # +1 for trailing '\n'
    			continue
		lines.append(line)

# shuffle passwords
# np.random.shuffle(lines)

# build vocabulary in decreasing order of occurrence
counter = collections.Counter(char for line in lines for char in line)
counts = sorted(counter.items(), key=lambda x: -x[1])

tokens, _ = zip(*counts)
vocab_size = len(tokens)

vocab = dict(zip(tokens, range(vocab_size)))

with open(vocab_file, 'wb') as f:
	pickle.dump(vocab, f)

# split into train/valid/test
total_size = len(lines)
valid_size = int(args.valid_frac * total_size)
test_size = int(args.test_frac * total_size)
train_size = total_size - valid_size - test_size

train_lines = lines[:train_size]
valid_lines = lines[train_size:train_size + valid_size]
test_lines = lines[train_size + valid_size:]

with open(train_file, 'w') as f:
	f.writelines(train_lines)

with open(valid_file, 'w') as f:
	f.writelines(valid_lines)

with open(test_file, 'w') as f:
	f.writelines(test_lines)

if args.verbose:
    print("preprocess done")
    print("vocab size: {}, total size: {}, train size: {}, validation size: {}, test size {}".format(vocab_size, total_size, train_size, valid_size, test_size))
    print("vocab and frequency rank: {}".format(vocab))
