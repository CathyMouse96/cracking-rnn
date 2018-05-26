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
	parser.add_argument('--sample_size', type=float, default='10000', \
	help='size of password sample (default: 10000)')
	parser.add_argument('--min_length', type=int, default='6', \
	help='minimum length of passwords to output (default: 6)')
	parser.add_argument('--max_length', type=int, default='12', \
	help='maximum length of passwords to output (default: 12)')
        parser.add_argument('--output_file', type=str, default='sampleprobs.txt', \
        help='file to store sampled passwords')
	parser.add_argument('--display_every', type=int, default=500, \
        help='display frequency (default: 500)')
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

	lut = {}

	vocab = loader.vocab
	charset = vocab.keys()
	charset_ordered = sorted(vocab.keys(), key=(lambda key: vocab[key]))

	results = []
	results_len = 0

	# Load first character probabilities
	first_char_probs = loader.first_char_probs
	for c in charset:
		if first_char_probs.has_key(c):
			if vocab[c] == 0:
				continue
			else:
				lut[c] = first_char_probs[c]

	total_start = time.time()

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables())
		ckpt = tf.train.get_checkpoint_state(args.save_dir)
        	if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
		start = time.time()
		while results_len < args.sample_size:
			# Pick first letter according to probability
			first_char = np.random.choice(first_char_probs.keys(), p = first_char_probs.values())
			# print first_char + '\t' + str(first_char_probs[first_char])
			current_prefix = first_char
			result_prob = first_char_probs[first_char]
			err = False
			while not current_prefix.endswith("\n"):
				# Pick next letter according to probability
				length = len(current_prefix)
				# Get next possible characters' probabilities by NN
                        	line = np.array(map(vocab.get, current_prefix))
                        	line = np.pad(line, (0, saved_args.seq_length - len(line)), 'constant')
                        	feed = {
                                	model.input_data: [line],
                                	model.sequence_lengths: [length]
                        	}
                        	probs = sess.run([model.probs], feed)
                        	probs = np.reshape(probs, (-1, saved_args.vocab_size))
                        	next_char_prob = probs[length - 1]
				# next_char_prob[i] is probability of char in vocab with value i
				next_char = np.random.choice(charset_ordered, p = next_char_prob)
				# print next_char + '\t' + str(next_char_prob[vocab[next_char]])
				current_prefix += next_char
				result_prob *= next_char_prob[vocab[next_char]]
				if len(current_prefix) > saved_args.seq_length: # this shouldn't happen
					print "Something that shouldn't happen happened"
					err = True
					break
			if err:
				continue
			# print str(result_prob) + '\t' + current_prefix
			results.append(str(result_prob) + '\n')
			results_len += 1
			if results_len % args.display_every == 0:
				end = time.time()
				print("Progress: {}/{}; time taken = {}".format(results_len, args.sample_size, end - start))
				start = time.time()

	with open(args.output_file, 'w') as f:
		f.writelines(results)
	total_end = time.time()
	print("Generated {} samples; total time taken = {}".format(len(results), total_end - total_start))

if __name__ == '__main__':
	main()

