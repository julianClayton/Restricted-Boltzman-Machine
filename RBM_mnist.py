import tensorflow as tf
import numpy as np
import data_reader 
import matplotlib.pyplot as plt
from random import randint
import math

class RBM:


	def __init__(
		self,
		input_layer_n,
		hidden_layer_n,
		iterations,
		dataset,
		batch_size=50,
		alpha=0.01):

		self.images = []
		self.batch_size = batch_size
		self.display_step = 3
		self.iterations = iterations
		self.dataset = dataset

		num_examples = dataset.num_examples
		

		self.x  = tf.placeholder(tf.float32, [None, input_layer_n], name="x")

		with tf.name_scope("weights") as scope: 
			W  = tf.Variable(tf.random_normal([input_layer_n, hidden_layer_n], 0.01), name="W") 

		with tf.name_scope("hidden_biases"):
			bh = tf.Variable(tf.random_normal([hidden_layer_n], 0.01),  tf.float32, name="bh")

		with tf.name_scope("visible_biases"):
			bv = tf.Variable(tf.random_normal([input_layer_n], 0.01),  tf.float32, name="bv")


		#first pass from input to hidden
		with tf.name_scope("hidden_nodes_0") as scope:
			hidden_activation = tf.nn.sigmoid(tf.matmul(self.x, W) + bh)
			hidden_0 		  = self.sample_probabilities(hidden_activation)


		#first pass back to visible
		with tf.name_scope("visible_nodes_1") as scope:
			visible_activation = tf.nn.sigmoid(tf.matmul(hidden_0, tf.transpose(W)) + bv)
			visible_1  		   = self.sample_probabilities(visible_activation)

		#second pass back to hidden 

		with tf.name_scope("hidden_nodes_1") as scope:
			hidden_1 = tf.nn.sigmoid(tf.matmul(visible_1, W) + bh)


		#Gradients for weights
		self.postive_grad = tf.matmul(tf.transpose(self.x), hidden_0)
		self.negative_grad = tf.matmul(tf.transpose(visible_1), hidden_1)

		with tf.name_scope("train") as scope:
			update_weights = alpha * tf.subtract(self.postive_grad, self.negative_grad)
			update_bv = alpha * tf.reduce_mean(tf.subtract(self.x, visible_1), 0)
			update_bh = alpha * tf.reduce_mean(tf.subtract(hidden_0 , hidden_1), 0)

		
		with tf.name_scope("hidden_sample") as scope:
			h_sample = self.sample_probabilities(tf.nn.sigmoid(tf.matmul(self.x, W) + bh))

		with tf.name_scope("visible_sample") as scope:
			self.v_sample = self.sample_probabilities(tf.nn.sigmoid(tf.matmul(h_sample, tf.transpose(W)) + bv))
			self.v = tf.nn.sigmoid(tf.matmul(h_sample, tf.transpose(W)) + bv)

		with tf.name_scope("objective_function") as scope:
			self.err = tf.reduce_mean(tf.subtract(self.x, self.v_sample)**2)
			tf.summary.scalar("objective_function", self.err)


		with tf.name_scope("update_all"):
			self.update_all = [W.assign_add(update_weights), bv.assign_add(update_bv), bh.assign_add(update_bh)]


		self.init  = tf.global_variables_initializer()
		self.merged_summary_op = tf.summary.merge_all()

	def sample_probabilities(self, probs):
		return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

	def draw_images(self):
		columns = 10
		rows = math.floor(len(self.images) / 10)
		iteration_counter = 0
		for i in range(len(self.images)):
			plt.subplot(rows, columns, i+1)
			plt.imshow(self.images[i], cmap='gray')
			if (i % 2 ==0):		#plots the input image: Before
				plt.title("B: " + str(iteration_counter))
			else:				#plots the output image: After 
				plt.title("A: " + str(iteration_counter))
				iteration_counter+=1
			plt.rcParams["axes.titlesize"] = 8
			plt.axis('off')
		plt.savefig('images/plot.png')
		plt.show()


	def run(self):
		with tf.Session() as sess:
			sess.run(self.init)
			summary_writer = tf.summary.FileWriter('data/logs', graph=sess.graph)
			for iteration in range(self.iterations):
				avg_cost = 0
				num_batches = int(self.dataset.num_examples/self.batch_size)
				for i in range(num_batches):
					batch_xs, _ = self.dataset.next_batch(self.batch_size)
					sess.run(self.update_all, feed_dict={self.x: batch_xs})
					avg_cost += sess.run(self.err, feed_dict={self.x: batch_xs})/num_batches
					summary_str = sess.run(self.merged_summary_op, feed_dict={self.x: batch_xs})
					summary_writer.add_summary(summary_str, iteration*num_batches + i)

				#Get before and after images and append them to the images array to be saved later
				#Will show a random before and after of an image from each epoch
				rand = randint(0, self.batch_size-1)
				before_im = sess.run(self.x, feed_dict={self.x: batch_xs})
				grey = before_im[rand, :].reshape((28,28))
				self.images.append(grey)
				after_im = sess.run(self.v, feed_dict={self.x: batch_xs})
				grey = after_im[rand, :].reshape((28,28))
				self.images.append(grey)

				if iteration % self.display_step == 0:
					print("iteration: " + str(iteration)+  " /" + str(self.iterations) + " COST: " + str(avg_cost))

			self.draw_images()

if __name__ == "__main__":
	training_data = data_reader.MNIST()
	rbm = RBM(784,200,30 , training_data)
	rbm.run()