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
	parser.add_argument('--save_dir', type=str, default='save', \
	help='model directory to store checkpointed models')
	parser.add_argument('--threshold', type=float, default='0.000001', \
	help='threshold for generating passwords (default: 10e-6)')
	parser.add_argument('--min_length', type=int, default='6', \
	help='minimum length of passwords to output (default: 6)')
        parser.add_argument('--output_file', type=str, default='output.txt', \
        help='file to store sampled passwords')
	args = parser.parse_args()
	return args

def main():
	args = parse_args()
	with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
		saved_args = pickle.load(f)
	loader = TextLoader(saved_args.data_dir, saved_args.batch_size, \
	saved_args.seq_length, isTraining=False)

	saved_args.batch_size = 1 # Set batch size to 1 when sampling
	model = Model(saved_args, training=False)

	prefixes = []
	lut = {}

	vocab = loader.vocab
	charset = vocab.keys()

	results = []

	# Load first character probabilities
	first_char_probs = loader.first_char_probs
	for c in charset:
		if first_char_probs.has_key(c) and first_char_probs[c] > args.threshold:
			if vocab[c] == 0:
				continue
			else:
				prefixes.append(c)
				lut[c] = first_char_probs[c]
	
	print prefixes

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables())
		ckpt = tf.train.get_checkpoint_state(args.save_dir)
        	if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			# Iterate all possible passwords probability more than threshold
			while prefixes: # while not empty
				current_prefix = prefixes.pop()
				start = time.time()
				length = len(current_prefix)
				line = np.array(map(vocab.get, current_prefix))
				line = np.pad(line, (0, saved_args.seq_length - len(line)), 'constant')
				# Get next possible characters' probabilities by NN
				feed = {
					model.input_data: [line],
					model.sequence_lengths: [length]
				}
				probs = sess.run([model.probs], feed)
				probs = np.reshape(probs, (-1, saved_args.vocab_size))
				next_char_prob = probs[length]
				for c in charset:
    					result_prob = lut[current_prefix] * next_char_prob[vocab[c]]
					result_str = current_prefix + c
					if result_prob > args.threshold:
    						if vocab[c] == 0:
    							if len(result_str) >= args.min_length:
    								results.append(result_str)
						else:
    							prefixes.append(result_str)
							lut[result_str] = result_prob
				end = time.time()
				print("sampled with prefix {}, time elapsed = {}".format(current_prefix, end - start))
	with open(output_file, 'w') as f:
		f.writelines(results)

if __name__ == '__main__':
	main()
