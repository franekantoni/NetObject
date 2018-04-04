F.Szombara

Implementation of Neural Net object using Python 3.6 and Tensorflow 1.6.0.
Allows for faster building and training simple feed forward neural networks.
Feel free to use the code however you want if you find it helpful.

NetObject(self, layers_nodes, input_size, output_size):
	
	Creates layers of a neural in a form 
	of dictionary where keys are the names of each layer in the format:
	hidden_lX, where X is the numer of a layer
	eg. "hidden_l1"- key of the first hidden layer
		"hidden_l0"- key of the input layer
	each layer is represented as a dictionary where "weights" is the key of
	weights and "biases" is the key of biases

	Parameters:
	layers_nodes: a list of numbers of neurons for each layer of the net
	eg. [500,500,500]- 3 layers, 500 neuons each

	input_size: length of the input vector

	output_size: size of the output vector


feed(self, x):

	Defines and performs operations in the neural network 
	returns the output layer

	Parameters:

	x: tf placeholder for the data

	Reurns:
	output layer of neural net


fit(self, train_x, train_y, test_x, test_y, hm_epochs=10, save_path="",batch_size=100):

	Trains a neural network, prints the final accuracy score. Saves
	the network to an outside file if save_path parameter is provided

	* Parameters:

	train_x: feature vectors for training

	train_y: lable vectors for training 

	test_x: feature vectors for testing

	test_y: lable vectors for testing

	hm_epochs: how many epochs to train the net (default hm_epochs=10)

	save_path: if provided saves the model to a file outside the program,
	tf documentation: https://www.tensorflow.org/programmers_guide/saved_model#save_and_restore_models
	(default: save_path=""- network will not be saved)

	batch_size: size of training batch (default: batch_size=100)

