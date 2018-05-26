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
	parser.add_argument('--init_from', type=int, default=None, \
	help="initialize with stored value and begin at this index (default: None)")
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
	next_char_probs = {}

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

		begin_index = 0
		if args.init_from != None:
			lut_file = os.path.join(args.save_dir, 'lut.pkl')
			with open(lut_file, 'rb') as f:
				lut = pickle.load(f)
			next_char_probs_file = os.path.join(args.save_dir, 'next_char_probs.pkl')
                        with open(next_char_probs_file, 'rb') as f:
                                next_char_probs = pickle.load(f)
			print("lut initialized from {}".format(lut_file))
			print("next_char_probs initialized from {}".format(next_char_probs_file))

			partial_results_file = args.output_file + "-" + str(args.init_from)
			with open(partial_results_file, 'r') as f:
				results = f.readlines()
			assert len(results) == args.init_from, "Unexpected error!"
			results_len = len(results)

			begin_index = args.init_from

		start = time.time()
		for testline in testset[begin_index:]:
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
    					# Try to get next possible characters' probabilities from dict
					if next_char_probs.has_key(current_prefix):
						# print "Found next char prob of <" + current_prefix + "> in dict!"
						next_char_prob = next_char_probs[current_prefix]
					# Otherwise get next possible characters' probabilities by NN
					else:
						length = len(current_prefix)
                                		line = np.array(map(vocab.get, current_prefix))
                                		line = np.pad(line, (0, saved_args.seq_length - len(line)), 'constant')
                                		feed = {
                                        		model.input_data: [line],
                                        		model.sequence_lengths: [length]
                                		}
                                		probs = sess.run([model.probs], feed)
                                		probs = np.reshape(probs, (-1, saved_args.vocab_size))
                                		next_char_prob = probs[length - 1]
						# Add next possible characters' probabilities to dict
						next_char_probs[current_prefix] = next_char_prob

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
				next_char_probs_file = os.path.join(args.save_dir, 'next_char_probs.pkl')
				with open(next_char_probs_file, 'wb') as f:
					pickle.dump(next_char_probs, f)
				print("lut saved to {}".format(lut_file))
				print("next_char_probs saved to {}".format(next_char_probs_file))

				partial_results_file = args.output_file + "-" + str(len(results))
				with open(partial_results_file, 'w') as f:
                			f.writelines(results)
				end = time.time()
				print("Written partial results to {}; time taken = {}".format(partial_results_file, end - start))
				start = time.time()

			if args.early_exit != None and results_len >= args.early_exit:
				break

        with open(args.output_file, 'w') as f:
                f.writelines(results)
        total_end = time.time()
        print("Finished assigning probabilities to {} passwords; total time taken = {}".format(len(results), total_end - total_start))

if __name__ == '__main__':
	main()

