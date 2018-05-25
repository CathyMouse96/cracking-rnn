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
        parser.add_argument('--test_file', type=str, default='preprocessed/test.txt', \
        help='file containing test set of passwords')
        parser.add_argument('--output_file', type=str, default='testprobs.txt', \
        help='file to store sampled passwords')
	parser.add_argument('--save_every', type=int, default=5000, \
	help='save frequency (default: 5000)')
	parser.add_argument('--display_every', type=int, default=500, \
        help='display frequency (default: 500)')
	parser.add_argument('--init_from', type=bool, default=False, \
	help="initialize lut with stored values (default: False)")
        parser.add_argument('--early_exit', type=int, default=None, \
        help='exit after assigning probabilities to this many passwords (default: None)')
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

	with open(args.test_file, 'r') as f:
		testset = f.readlines()
	testset_len = len(testset)

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

        with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver(tf.global_variables())
                ckpt = tf.train.get_checkpoint_state(args.save_dir)
                if ckpt and ckpt.model_checkpoint_path:
                        saver.restore(sess, ckpt.model_checkpoint_path)
		if args.init_from:
			lut_file = os.path.join(args.save_dir, 'lut.pkl')
			with open(lut_file, 'rb') as f:
				lut = pickle.load(f)
			print("lut initialized from {}".format(lut_file))

		start = time.time()
		for testline in testset:
			result_prob = 0.0
			# print "testline: " + testline
			# Find probability for existing prefix
			for k in range(1, len(testline)):
				if not lut.has_key(testline[:-k]):
					continue
				current_prefix = testline[:-k]
				# print "Found prefix in lut: " + current_prefix
				result_prob = lut[current_prefix]
				# Find probability for the rest of the string
				for m in range(k):
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

					next_char = testline[-k+m]
					current_prefix += next_char
					result_prob *= next_char_prob[vocab[next_char]]
					# Add new string to lut
					lut[current_prefix] = result_prob
				break
                        # print str(result_prob) + '\t' + current_prefix
                        results.append(str(result_prob) + '\n')
			results_len += 1
			if results_len % args.display_every == 0:
				end = time.time()
				print("Progress: {}/{}; time taken = {}".format(results_len, testset_len, end - start))
				start = time.time()
			if results_len % args.save_every == 0:
				lut_file = os.path.join(args.save_dir, 'lut.pkl')
				with open(lut_file, 'wb') as f:
					pickle.dump(lut, f)
				print("lut saved to {}".format(lut_file))
			if args.early_exit != None and results_len >= args.early_exit:
				break

        with open(args.output_file, 'w') as f:
                f.writelines(results)
        total_end = time.time()
        print("Finished assigning probabilities to {} passwords; total time taken = {}".format(len(results), total_end - total_start))

if __name__ == '__main__':
	main()

