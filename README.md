# deeplearning_assistant
For simple problem, many deep learning problem share many parts. Thus, implementing shared part in every time to solve those simple problems is very inefficient. So, this package provides a reusable code for typical parts of deep learning, and all user need is just implement own Neural Network Architecture and Iterator to feed the network.

You can inherit AbstractModelBuilder Class in [model_builder.py](https://github.com/kh-kim/deeplearning_assistant/blob/master/model_builder.py) to implement your own Network architecture, and also inherit [Iterator Class of Keras](https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py) to implement to feed the network.

After doing above tasks, set a hyper parameters like [hparams.json](https://github.com/kh-kim/deeplearning_assistant/blob/master/hparams.json). Also, more parameters are available and you can refer it in [assistant.py](https://github.com/kh-kim/deeplearning_assistant/blob/master/assistant.py).

Finally, after implement the code like [run.py](https://github.com/kh-kim/deeplearning_assistant/blob/master/run.py), you will get another version of deep learning training example!!
