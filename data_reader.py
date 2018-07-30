from tensorflow.examples.tutorials.mnist import input_data
import mnist_reader
import numpy

def MNIST():
	return MNISTData()

class Data:

	def __init__(self, training_X, training_y, testing_X, testing_y, num_examples, input_dim, output_dim):
		self._training_X = training_X
		self._training_y = training_y
		self._testing_X	= testing_X
		self._testing_y 	= testing_y
		self._num_examples = num_examples
		self._index_in_epoch = 0
		self._epochs_completed = 0
		self._input_dim = input_dim
		self._output_dim = output_dim

	def next_batch(self, batch_size):
		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self._num_examples:
			self._epochs_completed += 1
			perm = numpy.arange(self._num_examples)
			numpy.random.shuffle(perm)
			self._training_X = self._training_X[perm]
			self._training_y = self._training_y[perm]
			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self.num_examples
		end = self._index_in_epoch
		return self._training_X[start:end], self._training_y[start:end]

	@property
	def training_X(self):
		return self._training_X

	@property
	def training_y(self):
		return self._training_y

	@property
	def testing_X(self):
		return self._testing_X

	@property
	def testing_y(self):
		return self._testing_y

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def output_dim(self):
		return self._output_dim
	
	@property
	def input_dim(self):
		return self._input_dim

	
	
class MNISTData(Data):

	def __init__(self):
		training_X, training_y, testing_X, testing_y  = mnist_reader.read_data_sets()
		Data.__init__(self,  training_X, training_y, testing_X, testing_y, 55000, 784, 10)

if __name__ == "__main__":

	poker_hand()
 