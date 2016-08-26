import numpy as np

from model_builder import AbstractModelBuilder
from assistant import DeepLearningAssistant
from simple_iterator import SimpleIterator

class MyModelBuilder(AbstractModelBuilder):

	def buildModel(self):	# You cannot modify the method signature.
		# Instead changing method signature directly, you can modify __init__ to add class variable, and call (use) it inside of this method.
		from keras.models import Sequential, Model
		from keras.layers.core import Dense, Dropout
		
		model = Sequential()
		model.add(Dense(100, input_dim = 10, activation = 'relu'))
		model.add(Dense(80, input_dim = 10, activation = 'relu'))
		model.add(Dense(50, input_dim = 10, activation = 'relu'))
		model.add(Dense(5, activation = 'sigmoid'))
		
		# You have to return model.
		return model

if __name__ == "__main__":
	# Our deep learning assistant needs hyper parameters for training. You can pass it as JSON file.
	assistant = DeepLearningAssistant("./hparams.json")
	# Init your own Neural Network Architecture to solve your problem.
	mb = MyModelBuilder()
	# In this simple example, "SimpleIterator" class is used for iterator, which iterates small data-set on the memory.
	# However, if the data-set is too big to load to memory at once, you need to implement your own Iterator instead of using this.
	iterator = SimpleIterator(assistant.BATCH_SIZE)

	# Pass model builder with training iterator and test iterator
	# In this example, we just use same iterator for training iterator and test iterator. 
	# However, note that you need separate two data-set (iterator) in real implementation!!
	assistant.train(mb, iterator, iterator)