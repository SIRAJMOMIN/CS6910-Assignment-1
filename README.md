   FeedForward Neural Network Implementation.
   
   TYPE OF PROJECT- Deep Learning
   
   1stGOAL OF PROJECT- 1.Implement and use gradient descent and other optimizer algorithms such has adam , nadam , sgd , mgd ,nesterov ,rmsprop
   
   2ndGOAL OF PROJECT -2.To bulit a feed forward neural network and back propagation algorithm which can work fine for various size of experiments and independent on count of number of hidden layers

   SINGLE_LAYER - The layer consissts of various activation functions and optimizer algorithms

   In each pass we send parameters with initialization of weights and other parameters 

   All the activation functions-softmax,tanh,relu,sigmoid ae defined 

   All the optimizer algorithms for a less loss benefit are defined in this layer -gd,sgd,mgd,nesterov,adam,nadam,rmsprop

   All these various functions are used while training the network model

   FEED_FORWARD_NEURAL_NETWORK-1.epochs : Number of times we want to run the algorithm on whole dataset (default:5)

   2.hidden_size : Number of neurons in a feedforwardneuralnetworklayer

   3.learn_rate : Learning rate used to optimize model parameters (default = 0.01)

   4.hidden_layer:Number of hidden layers used in feedforward neural network -batch_size: Batch size to train the dataset (default = 64) 

   5.activation :Activation function used in feedforward neural network(default = "sigmoid") (options = "relu", "sigmoid", "tanh")

   6.optimizer : Optimizer used to train neural network (default = "gradient_descent") (options = "gradient_descent" , "stochastic_gradient_descent", "momentum_gradient_descent", "nesterov" ,"rmsprop", "adam", "nadam")

   7.l2_lambda :Weight decay used by optimizers (default = 0)

   8.weight_initial : Weight initialization method used to initialize neural network weights (default = "random") (options = "random", "xavier" )

   9.loss : Loss function used to train neural network (default = "cross_entropy") (options = "cross_entropy" , "mean_square")

   FORWARD_PROPAGATE-This function passes the input X through all the layers in the neural network in the forward direction and returns the output of the last layer.

   BACKWARD_PROPAGATE-This function passes the true output Y and predicted output A through all the layers in the neural network in the backward direction and computes the gradients of the loss function with respect to the weights

   FIT_FUNCTION -This function trains the neural network on the training data, and computes the loss for the validation and test data. 
   
   Generates one-hot encoded vectors for the training, validation, and test data, and updates the learning rate for the ReLU activation function and certain optimizers.

   It also splits the training data into training and validation sets, adds the layers to the neural network.

   TRAINING THE MODEL-To train the model we need to initalize the Neural network with NeuralNetwork() function which calls the layer() function and Train the model using fit() method

   MODEL EVALUATION -To evaluate the model just pass the test data to fit() function it will display the validation loss, validation accuracy, test loss and test accuracy

   

  









                  

   
