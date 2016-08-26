from model_builder import AbstractModelBuilder

class SampleModelBuilder(AbstractModelBuilder):

	def buildModel(self):	#You cannot modify the method signature.
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
