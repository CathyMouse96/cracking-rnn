import argparse
import time
import os
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

from utils import TextLoader
from model import Model

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch-size', type=int, default=64, \
	help='batch size (default: 64)')
	parser.add_argument('--seq_length', type=int, default=16, \
	help='RNN sequence length (default: 16)')
	parser.add_argument('--num_epochs', type=int, default=50, \
	help='number of epochs (default: 50)')
	parser.add_argument('--vocab_size', type=int, default=None, \
	help="vocabulary size (default: infer from input)")
	parser.add_argument('--learning_rate', type=float, default=0.002, \
	help='learning rate (default: 0.002)')
	parser.add_argument('--save_every', type=int, default=1000, \
	help='save frequency (default: 1000)')
	parser.add_argument('--display_every', type=int, default=100, \
	help='display frequency (default: 100)')
	parser.add_argument('--rnn_size', type=int, default=128, \
	help='size of RNN hidden state (default: 128)')
	parser.add_argument('--data_dir', type=str, default='preprocessed', 
	help='data directory containing input (default: preprocessed)')
	parser.add_argument('--save_dir', type=str, default='save', \
	help='directory to store checkpointed models (default: save)')
	
	args = parser.parse_args()
	return args

def main():
	args = parse_args()
	loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)
	args.vocab_size = loader.vocab_size
	print("vocab_size = {}".format(args.vocab_size))

	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)
	with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
		pickle.dump(args, f)
	with open(os.path.join(args.save_dir, 'vocab.pkl'), 'wb') as f:
		pickle.dump(loader.vocab, f)
	
	model = Model(args)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables())

		for e in range(args.num_epochs):
			loader.reset_batch_pointer()
			for b in range(loader.num_batches):
				start = time.time()
				x, y, length = loader.next_batch()
				feed = {
					model.input_data: x,
					model.targets: y,
					model.sequence_lengths: length
				}
				train_loss, _ = sess.run([model.cost, model.optimizer], feed)
				end = time.time()
				global_step = e * loader.num_batches + b
				if global_step % args.display_every == 0 and global_step != 0:
					print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
					.format(b, loader.num_batches, e, train_loss, end - start))
				if global_step % args.save_every == 0 and global_step != 0:
					checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
					saver.save(sess, checkpoint_path, global_step=global_step)
					print("model saved to {}".format(checkpoint_path))

if __name__ == '__main__':
    main()
