import tensorflow as tf
import numpy as np
import data_reader 
import matplotlib.pyplot as plt
from random import randint


class RBM:


	def __init__(
		self,
		input_layer_n,
		hidden_layer_n,
		iterations,
		dataset,
		batch_size=50):

		self.batch_size = batch_size
		self.display_step = 3
		self.iterations = iterations
		self.dataset = dataset
		num_examples = dataset.num_examples
		alpha = 0.01

		self.x  = tf.placeholder(tf.float32, [None, input_layer_n], name="x") 
		W  = tf.Variable(tf.random_normal([input_layer_n, hidden_layer_n], 0.01), name="W") 
		bh = tf.Variable(tf.random_normal([hidden_layer_n], 0.01),  tf.float32, name="bh")
		bv = tf.Variable(tf.random_normal([input_layer_n], 0.01),  tf.float32, name="bv")


		#first pass from input to hidden
		hidden_activation = tf.nn.sigmoid(tf.matmul(self.x, W) + bh)
		hidden_0 		  = self.get_activations(hidden_activation)


		#first pass back to visible
		visible_activation = tf.nn.sigmoid(tf.matmul(hidden_0, tf.transpose(W)) + bv)
		visible_1  		   = self.get_activations(visible_activation)

		#second pass back to hidden 

		hidden_1 = tf.nn.sigmoid(tf.matmul(visible_1, W) + bh)


		#Gradients for weights
		self.postive_grad = tf.matmul(tf.transpose(self.x), hidden_0)
		self.negative_grad = tf.matmul(tf.transpose(visible_1), hidden_1)

		update_weights = alpha * tf.subtract(self.postive_grad, self.negative_grad)
		update_bv = alpha * tf.reduce_mean(tf.subtract(self.x, visible_1), 0)
		update_bh = alpha * tf.reduce_mean(tf.subtract(hidden_0 , hidden_1), 0)

		h_sample = self.get_activations(tf.nn.sigmoid(tf.matmul(self.x, W) + bh))
		self.v_sample = self.get_activations(tf.nn.sigmoid(tf.matmul(h_sample, tf.transpose(W)) + bv))

		self.v = tf.nn.sigmoid(tf.matmul(h_sample, tf.transpose(W)) + bv)
		self.err = tf.reduce_mean((self.x - self.v_sample)**2)

		self.update_all = [W.assign_add(update_weights), bv.assign_add(update_bv), bh.assign_add(update_bh)]


		self.init  = tf.global_variables_initializer()

	def get_activations(self, probs):
		return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))


	def run(self):
		with tf.Session() as sess:
			sess.run(self.init)
			for iteration in range(self.iterations):
				avg_cost = 0
				num_batches = int(self.dataset.num_examples/self.batch_size)
				for i in range(num_batches):
					batch_xs, batch_ys = self.dataset.next_batch(self.batch_size)
					sess.run(self.update_all, feed_dict={self.x: batch_xs})
					avg_cost += sess.run(self.err, feed_dict={self.x: batch_xs})/num_batches
				if iteration % self.display_step == 0:
					print("iteration: " + str(iteration)+  " /" + str(self.iterations) + " COST: " + str(avg_cost))
				if iteration % 2 == 0:
					#This is how images are displayed for now. Reconstruction followed by the actual image
					rand = randint(0, self.batch_size)
					image = sess.run(self.v, feed_dict={self.x: batch_xs})
					grey = image[rand, :].reshape((28,28))
					plt.imshow(grey)
					plt.draw()
					plt.pause(2)
					plt.close()
					image2 = sess.run(self.x, feed_dict={self.x: batch_xs})
					grey = image2[rand, :].reshape((28,28))
					plt.imshow(grey)
					plt.draw()
					plt.pause(2)
					plt.close()

if __name__ == "__main__":
	training_data = data_reader.MNIST()
	rbm = RBM(784,200,60 , training_data)
	rbm.run()