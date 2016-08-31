import json
import numpy as np
import h5py

from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping

class DeepLearningAssistant:

	def __init__(self, settting_filename):		
		f = open(settting_filename, "r")
		js = json.loads(f.read())
		f.close()

		self.BACK_END = js.get("BACK_END") if js.get("BACK_END") != None else "th"
		self.BATCH_SIZE = int(js.get("BATCH_SIZE")) if js.get("BATCH_SIZE") != None else 128
		self.OPTIMIZER = js.get("OPTIMIZER") if js.get("OPTIMIZER") != None else "sgd"
		self.LOSS = js.get("LOSS") if js.get("LOSS") != None else "mse"
		self.MAX_EPOCH = int(js.get("MAX_EPOCH")) if js.get("MAX_EPOCH") != None else 100
		self.SAMPLES_PER_EPOCH = int(js.get("SAMPLES_PER_EPOCH")) if js.get("SAMPLES_PER_EPOCH") != None else -1
		self.LEARNING_RATE = float(js.get("LEARNING_RATE")) if js.get("LEARNING_RATE") != None else 0.025
		self.SAMPLES_PER_TEST = int(js.get("SAMPLES_PER_TEST")) if js.get("SAMPLES_PER_TEST") != None else -1

	def train(self, model_builder, train_iterator, test_iterator, verbose = 1, model_check_point = None, early_stopping = None):		
		self.modelBuilder = model_builder
		self.trainIterator = train_iterator
		self.testIterator = test_iterator

		model = self.modelBuilder.getModel()
		trainGenerator = self.trainIterator
		testGenerator = self.testIterator
		callbacks = []
		if model_check_point != None:
			callbacks.append(model_check_point)
		if early_stopping != None:
			callbacks.append(early_stopping)

		if self.SAMPLES_PER_EPOCH == -1:
			self.SAMPLES_PER_EPOCH = train_iterator.N
		if self.SAMPLES_PER_TEST == -1:
			self.SAMPLES_PER_TEST = test_iterator.N

		print "BACK_END: %s" % self.BACK_END
		print "LOSS: %s" % self.LOSS
		print "OPTIMIZER: %s" % self.OPTIMIZER

		if self.OPTIMIZER == "sgd":
			sgd = SGD(lr=self.LEARNING_RATE, decay=1e-6, momentum=0.9, nesterov=True)
			self.OPTIMIZER = sgd

			print "LEARNING RATE: %f" % self.LEARNING_RATE

		print ""
		print "BATCH_SIZE: %d" % self.BATCH_SIZE
		print "SAMPLES PER EPOCH: %d" % self.SAMPLES_PER_EPOCH
		print "SAMPLES PER TEST: %d" % self.SAMPLES_PER_TEST
		print "MAX EPOCH: %d" % self.MAX_EPOCH		

		model.compile(optimizer=self.OPTIMIZER, loss=self.LOSS, metrics=['accuracy'])

		history = model.fit_generator(trainGenerator,
				samples_per_epoch = self.SAMPLES_PER_EPOCH,
				nb_epoch = self.MAX_EPOCH,
				validation_data = testGenerator,
				nb_val_samples = self.SAMPLES_PER_TEST,
				verbose = verbose,
				callbacks = callbacks
				)

		return model
	
	def test(self):
		pass
