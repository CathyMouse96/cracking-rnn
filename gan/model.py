import os
import time
import numpy as np
import tensorflow as tf

class Model():
	def __init__(self, args, training=True):
		self.args = args
		self.sequence_lengths = tf.placeholder(tf.int32, shape=[self.args.batch_size])
		sequence_lengths = self.sequence_lengths

		self.real_inputs_discrete = tf.placeholder(tf.int32, shape=[self.args.batch_size, self.args.seq_length])
		real_inputs_discrete = self.real_inputs_discrete
		real_inputs = tf.one_hot(real_inputs_discrete, self.args.vocab_size)
		with tf.variable_scope("Generator"):
			fake_inputs = self.Generator(sequence_lengths)
		fake_inputs_discrete = tf.argmax(fake_inputs, fake_inputs.get_shape().ndims - 1)

		with tf.variable_scope("Discriminator"):
			disc_real = self.Discriminator(real_inputs, sequence_lengths)
		with tf.variable_scope("Discriminator", reuse=True):
			disc_fake = self.Discriminator(fake_inputs, sequence_lengths)

		disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
		gen_cost = -tf.reduce_mean(disc_fake)
		self.gen_cost = gen_cost

		# WGAN lipschitz-penalty
		alpha = tf.random_uniform(shape=[self.args.batch_size, 1, 1], minval=0., maxval=1.)
		differences = fake_inputs - real_inputs
		interpolates = real_inputs + (alpha * differences)
		with tf.variable_scope("Discriminator", reuse=True):
			gradients = tf.gradients(self.Discriminator(interpolates, sequence_lengths), [interpolates])[0]
		slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
		gradient_penalty = tf.reduce_mean((slopes-1.)**2)
		disc_cost += 10 * gradient_penalty # Gradient penalty lambda hyperparameter.
		self.disc_cost = disc_cost

		gen_params = [v for v in tf.global_variables() if v.name.startswith("Generator")]
		disc_params = [v for v in tf.global_variables() if v.name.startswith("Discriminator")]
		
		self.gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
		self.disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)


	def make_noise(self, shape):
		return tf.random_normal(shape)

	def Generator(self, sequence_lengths):
		inputs = self.make_noise(shape=[self.args.seq_length, self.args.batch_size, self.args.rnn_size])
		# Unstack inputs into list of 2D tensors
		inputs = tf.unstack(inputs, axis=0)

		# Build RNN cell
		cell_fn = tf.nn.rnn_cell.BasicLSTMCell
		cells = []
		for _ in range(self.args.num_layers):
    			cells.append(cell_fn(self.args.rnn_size))
		cell = tf.nn.rnn_cell.MultiRNNCell(cells)

		initial_state = cell.zero_state(self.args.batch_size, tf.float32)

		# Run Static RNN
		outputs, _ = tf.nn.static_rnn(cell, inputs, initial_state=None, dtype=tf.float32)

		# Perform softmax at output
		weights = tf.get_variable("weights", [self.args.rnn_size, self.args.vocab_size])
		bias = tf.get_variable("bias", [self.args.vocab_size])

		outputs = tf.reshape(outputs, [-1, self.args.rnn_size])
		logits = tf.split(tf.matmul(outputs, weights) + bias, self.args.batch_size, axis=0)
		probs = tf.nn.softmax(logits)

		return probs

	def Discriminator(self, inputs, sequence_lengths):
    		# Inputs are batch-majored. Transpose inputs to time-majored
		inputs = tf.transpose(inputs, [1, 0, 2])
		# Unstack inputs into list of 2D tensors
		inputs = tf.unstack(inputs, axis=0)

		# Build RNN cell
		disc_cell = tf.nn.rnn_cell.BasicLSTMCell(self.args.rnn_size, reuse=tf.get_variable_scope().reuse)
			
		disc_initial_state = disc_cell.zero_state(self.args.batch_size, tf.float32)

		# Run Static RNN
		_, state = tf.nn.static_rnn(disc_cell, inputs, initial_state=None, dtype=tf.float32)

		h_state = state[1]

		# Perform binary logistic regression at output
		weights = tf.get_variable("weights", [self.args.rnn_size, 2])
		bias = tf.get_variable("bias", [2])

		logits = tf.matmul(h_state, weights) + bias
		output = tf.nn.sigmoid(logits)

		return output

