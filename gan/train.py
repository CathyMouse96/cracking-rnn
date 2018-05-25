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
	parser.add_argument('--num_layers', type=int, default=1, \
	help='number of layers in the RNN (default: 1)')
	parser.add_argument('--data_dir', type=str, default='../preprocessed', \
	help='data directory containing input (default: preprocessed)')
	parser.add_argument('--save_dir', type=str, default='save', \
	help='directory to store checkpointed models (default: save)')
	parser.add_argument('--init_from', type=str, default=None, \
	help="checkpoint file or directory to intialize from (default: None)")
	
	args = parser.parse_args()
	return args

def main():
	args = parse_args()
	loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)
	args.vocab_size = loader.vocab_size
	print("vocab_size = {}".format(args.vocab_size))

	if args.init_from is not None:
    		if os.path.isdir(args.init_from): # init from directory
			assert os.path.exists(args.init_from), \
			"{} is not a directory".format(args.init_from)
			parent_dir = args.init_from
		else: # init from file
			assert os.path.exists("{}.index".format(args.init_from)), \
			"{} is not a checkpoint".format(args.init_from)
			parent_dir = os.path.dirname(args.init_from)

		config_file = os.path.join(parent_dir, 'config.pkl')
		vocab_file = os.path.join(parent_dir, 'vocab.pkl')

		assert os.path.isfile(config_file), \
		"config.pkl does not exist in directory {}".format(parent_dir)
		assert os.path.isfile(vocab_file), \
		"vocab.pkl does not exist in directory {}".format(parent_dir)

		if os.path.isdir(args.init_from):
			checkpoint = tf.train.latest_checkpoint(parent_dir)
			assert checkpoint, \
			"no checkpoint in directory {}".format(init_from)
		else:
			checkpoint = args.init_from

		with open(os.path.join(parent_dir, 'config.pkl'), 'rb') as f:
			saved_args = pickle.load(f)
		with open(os.path.join(parent_dir, 'vocab.pkl'), 'rb') as f:
			saved_vocab = pickle.load(f)
		assert saved_vocab == loader.vocab, \
		"vocab in data directory differs from save"

	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

	new_config_file = os.path.join(args.save_dir, 'config.pkl')
	new_vocab_file = os.path.join(args.save_dir, 'vocab.pkl')

	if not os.path.exists(new_config_file):
		with open(new_config_file, 'wb') as f:
			pickle.dump(args, f)
	if not os.path.exists(new_vocab_file):
		with open(new_vocab_file, 'wb') as f:
			pickle.dump(loader.vocab, f)
	
	model = Model(args)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables())

		if args.init_from is not None:
			try:
				saver.restore(sess, checkpoint)
			except ValueError:
				print("{} is not a valid checkpoint".format(checkpoint))
			print("initializing from {}".format(checkpoint))

		for e in range(args.num_epochs):
			loader.reset_batch_pointer()
			for b in range(loader.num_batches):

				start = time.time()
				
				x, _, length = loader.next_batch()

				# Train critic
				for i in xrange(1): # How many critic iterations per generator iteration.
					disc_feed = {
						model.real_inputs_discrete: x,
						model.sequence_lengths: length
					}
					disc_cost, _ = sess.run([model.disc_cost, model.disc_train_op], disc_feed)

				# Train generator
				gen_feed = {
					model.sequence_lengths: length
				}
				gen_cost, _ = sess.run([model.gen_cost, model.gen_train_op], gen_feed)

				end = time.time()

				global_step = e * loader.num_batches + b

				if global_step % args.display_every == 0 and global_step != 0:
					print("{}/{} (epoch {}), gen_cost = {:.3f}, disc_cost = {:.3f}, time/batch = {:.3f}" \
					.format(b, loader.num_batches, e, gen_cost, disc_cost, end - start))

				if global_step % args.save_every == 0 and global_step != 0:
					checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
					saver.save(sess, checkpoint_path, global_step=global_step)
					print("model saved to {}".format(checkpoint_path))

if __name__ == '__main__':
    main()

