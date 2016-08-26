import numpy as np

from assistant import DeepLearningAssistant
from simple_iterator import SimpleIterator
from sample_model_builder import SampleModelBuilder

if __name__ == "__main__":
	# Our deep learning assistant needs hyper parameters for training. You can pass it as JSON file.
	assistant = DeepLearningAssistant("./hparams.json")
	# Init your own Neural Network Architecture to solve your problem.
	mb = SampleModelBuilder()
	# In this simple example, "SimpleIterator" class is used for iterator, which iterates small data-set on the memory.
	# However, if the data-set is too big to load to memory at once, you need to implement your own Iterator instead of using this.
	iterator = SimpleIterator(assistant.BATCH_SIZE)

	# Pass model builder with training iterator and test iterator
	# In this example, we just use same iterator for training iterator and test iterator. 
	# However, note that you need separate two data-set (iterator) in real implementation!!
	assistant.train(mb, iterator, iterator, verbose = 2)
