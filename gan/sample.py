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
        help='directory to store checkpointed models (default: save)')
	parser.add_argument('--threshold', type=int, default='1000000', \
        help='number of passwords to generate (default: 1000000)')
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

	vocab = loader.vocab
	inv_vocab = {v: k for k, v in vocab.iteritems()}

	results = []
	results_len = 0

	total_start = time.time()

        with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver(tf.global_variables())
                ckpt = tf.train.get_checkpoint_state(args.save_dir)
                if ckpt and ckpt.model_checkpoint_path:
                        saver.restore(sess, ckpt.model_checkpoint_path)
			while results_len < args.threshold:
				sample_inds= sess.run(model.fake_inputs_discrete)
				sample_ind = sample_inds[0]
				sample_char = []
				for i in sample_ind:
					sample_char.append(inv_vocab[i])
					if i == 0:
						break
				sample_str = ''.join(sample_char)
				results.append(sample_str)
				results_len += 1
	with open(args.output_file, 'w') as f:
                f.writelines(results)
        total_end = time.time()
        print("Generated {} samples; total time taken = {}".format(len(results), total_end - total_start))

if __name__ == '__main__':
        main()

