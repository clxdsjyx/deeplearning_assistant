import numpy as np
from keras.preprocessing.image import Iterator

# For more information about this Iterator class, you can refer the below url:
# https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
class SimpleIterator(Iterator):

	def __init__(self, batch_size, shuffle = True, seed = None):
		# Data generation for this simple example.
		# Our Neural Network will be trained to predict the XOR operation for two arrays.
		from random import random
		data = []
		for i in xrange(10000):
			x = [0] * 10
			y = [0] * 5

			for j in xrange(5):
				x[j] =int(random() + 0.5)
				x[5 + j] =int(random() + 0.5)
				y[j] = x[j]^x[5 + j]	# Generate label for XOR operation.

			data.append((np.array(x), np.array(y)))

		self.data = data		
		N = len(data)

		super(SimpleIterator, self).__init__(N, batch_size, shuffle, seed)

	def next(self):
		with self.lock:
			index_array, current_index, current_batch_size = next(self.index_generator)

		batch_x = []
		batch_y = []

		for i, j in enumerate(index_array):
			x, y = self.data[j]
			batch_x.append(x)
			batch_y.append(y)

		return np.array(batch_x), np.array(batch_y)