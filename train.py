

import argparse
import wandb
from wandb.keras import WandbCallback
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
import numpy as np
from keras.datasets import mnist
wandb.login(key='e25b96b795b0569d21a9cb8b7bb036678e2e5f27')
parser = argparse.ArgumentParser()
parser.add_argument('-wp' , '--wandb_project', help='Project name used to track experiments in Weights & Biases dashboard' , type=str, default='Deep Learning Assignment 1')
parser.add_argument('-we', '--wandb_entity' , help='Wandb Entity used to track experiments in the Weights & Biases dashboard.' , type=str, default='CS23M064')
parser.add_argument('-d', '--dataset', help='choices: ["mnist", "fashion_mnist"]', type=str, default='fashion_mnist',choices=["mnist", "fashion_mnist"])
parser.add_argument('-e', '--epochs', help="Number of epochs to train neural network.", type=int, default=10 ,choices=[5,10,])
parser.add_argument('-b', '--batch_size', help="Batch size used to train neural network.", type=int, default=64,choices=[16,32,64])
parser.add_argument('-l','--loss', help = 'choices: ["mean_squared_error", "cross_entropy"]' , type=str, default='cross_entropy',choices=["mean_square", "cross_entropy"])
parser.add_argument('-o', '--optimizer', help = 'choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]', type=str, default = 'nadam',choices= ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
parser.add_argument('-lr', '--learning_rate', help = 'Learning rate used to optimize model parameters', type=float, default=1e-03,choices=[1e-3, 1e-4] )
parser.add_argument('-m', '--momentum', help='Momentum used by momentum and nag optimizers.',type=float, default=0.5)
parser.add_argument('-beta', '--beta', help='Beta used by rmsprop optimizer',type=float, default=0.5)
parser.add_argument('-beta1', '--beta1', help='Beta1 used by adam and nadam optimizers.',type=float, default=0.5)
parser.add_argument('-beta2', '--beta2', help='Beta2 used by adam and nadam optimizers.',type=float, default=0.5)
parser.add_argument('-eps', '--epsilon', help='Epsilon used by optimizers.',type=float, default=0.000001)
parser.add_argument('-w_d', '--weight_decay', help='Weight decay used by optimizers.',type=float, default=.0,choices=[0, 0.0005,  0.5])
parser.add_argument('-w_i', '--weight_init', help = 'choices: ["random", "Xavier"]', type=str, default='random',choices=["random", "xavier"])
parser.add_argument('-nhl', '--num_layers', help='Number of hidden layers used in feedforward neural network.',type=int, default=4,choices=[3, 4, 5])
parser.add_argument('-sz', '--hidden_size', help ='Number of hidden neurons in a feedforward layer.', nargs='+', type=int, default=128,choices=[32, 64, 128], required=False)
parser.add_argument('-a', '--activation', help='choices: ["identity", "sigmoid", "tanh", "relu"]', type=str, default='relu',choices= ['sigmoid','tanh','relu'])
# parser.add_argument('--hlayer_size', type=int, default=32)
parser.add_argument('-oa', '--output_activation', help = 'choices: ["softmax"]', type=str, default='softmax')
# parser.add_argument('-oc', '--output_size', help ='Number of neurons in output layer used in feedforward neural network.', type = int, default = 10)
arguments = parser.parse_args()
wandb.init(project=arguments.wandb_project,entity=arguments.wandb_entity)

# Load the training and testing data
if(arguments.dataset=="fashion_mnist"):
    ((x_train,y_train),(x_test,y_test)) = fashion_mnist.load_data()
elif(arguments.dataset=="mnist"):
    ((x_train,y_train),(x_test,y_test)) = mnist.load_data()

class SingleLayer:
#creating a class with a layer as an object and parametes to the layer are :
    #input dimensions
    #activation_function
    #optimizer_function
    #weight initialization #QUESTION 2 AND QUESTION 3
# optimizer and activation functions
  def __init__(self, idim, nof_nodes, activation='', optimizer='gradient_descent', weight_type='random'):
    self.optimizer = self.do_optimizer(optimizer)
    self.opt=optimizer
    self.activation, self.activationForwardFunction, self.activationBackwardFunction = self.do_activation(activation)
# initializing Momentum and velocity weights and bias
    self.weights, self.bias = self.initialize(idim, nof_nodes, activation, weight_type=weight_type)
    self.pv_weight, self.pv_bias = np.zeros([nof_nodes, idim]), np.zeros([nof_nodes, 1])
    self.pm_weight, self.pm_bias = np.zeros([nof_nodes, idim]), np.zeros([nof_nodes, 1])
# initialization of weights and bias to layer:
# with random-normal distribution
# xavier distribution
  def initialize(self, nof_inputfeatures, nof_nodes,activation,weight_type):
    np.random.seed(1)
    if weight_type == 'random':
      w=np.random.normal(0.0,0.5,size=(nof_nodes, nof_inputfeatures))
    else:
      x=np.sqrt(nof_nodes)
      w=np.random.uniform(-(1/x), (1/x), size=(nof_nodes, nof_inputfeatures))
    b = np.ones([nof_nodes, 1])
    return w,b
# selection of optimizer based on input value
# gradient_descent
#momentum_gradient_descent ,sgd,nadam,adam,nesterov
#optimization function
  def do_optimizer(self, optimizer):
    if optimizer == 'gradient_descent':
        return self.gradient_descent
    elif optimizer == 'momentum':
        return self.momentum_gradient_descent
    elif optimizer == 'rmsprop':
        return self.rmsprop
    elif optimizer == 'adam':
        return self.adam
    elif optimizer == 'sgd':
        return self.stochastic_gradient_descent
    elif optimizer == 'nadam':
        return self.nadam
    elif optimizer == 'nag':
        return self.nesterov
#selection of activation function based on input value
# activation functions such has sigmoid ,relu and tanh and softmax
  def do_activation(self, activation):
    if activation == 'sigmoid':
        return activation, self.sigmoid, self.sigmoid_grad
    elif activation == 'relu':
        return activation, self.relu, self.relu_grad
    elif activation == 'tanh':
        return activation, self.tanh, self.tanh_grad
    else:
        return 'softmax', self.softmax, self.softmax_grad

#activation functions
#activation functions with their derivatives
#sigmoid function
  def sigmoid(self, Z):
    Z=np.clip(Z,500,-500)
    A = 1 / (1 + np.exp(-Z))
    return A

# derivative of sigmoid function
  def sigmoid_grad(self, derivative_A):
    e=np.exp(-(self.previous_Z))
    s = 1/(1+e)
    derivative_Z = derivative_A * s * (1 - s)
    return derivative_Z

 #tanh activation function
  def tanh(self,Z):
    return np.tanh(Z)
# derivative of tanh function
  def tanh_grad(self,derivative_A):
    s=self.tanh(self.previous_Z)
    ss=(s**2)
    return derivative_A*(1-ss)
#relu activation function
  def relu(self,Z):
    A= np.maximum(0,Z)
    return A
# derivative of relu function
  def relu_grad(self,derivative_A):
    s= np.maximum(0,self.previous_Z)
    t = 1.*(s>0)*derivative_A
    return t
#softmax activation function
  def softmax(self,Z):
    maxZ=np.max(Z)
    eZ=np.exp(Z - maxZ)
    A = eZ/eZ.sum(axis=0, keepdims=True)
    return A
#gradient of softmax function
  def softmax_grad(self,derivative_A):
    return derivative_A

#forward propagation of input vector A
  def forward_propagate(self, A):
    if self.opt != 'nesterov':
      Z= np.dot(self.weights,A) + self.bias
    else:
      xw=0.9*self.pv_weight
      xw=self.weights-xw
      xb=0.9*self.pv_bias
      xb=self.bias-xb
      Z=np.dot(xw, A) + xb
    self.previous_A = A
    self.previous_Z = Z
    A = self.activationForwardFunction(Z)
    return A
#backward propagation
  def backward_propagate(self, derivative_A):
    sp=self.previous_A.shape[1]
    derivative_Z = self.activationBackwardFunction(derivative_A)
    sum_value=np.sum(derivative_Z, axis=1, keepdims=True)
    self.derivative_b = 1/sp*sum_value
    self.derivative_w = 1 / sp * np.dot(derivative_Z, self.previous_A.T)
    return np.dot(self.weights.T, derivative_A)

  def predict(self,A):
    x=np.dot(self.weights,A)
    Z=self.bias + x
    A=self.activationForwardFunction(Z)
    return A

#stochastic gradient descent algorithm for updating weights
  def stochastic_gradient_descent(self, derivative_A,learn_rate = 0.001,t = 0,l2_lambda=0,batch_size = 32):
    derivative_Z=self.activationBackwardFunction(derivative_A)
    a= self.previous_A.shape[1]
    previous_derivative_A = np.dot(self.weights.T, derivative_Z)
    for i in range(a):
      b=derivative_Z[:,i:i+1]
      self.derivative_b=1/a*b
      self.derivative_w = 1/a*np.dot(b,self.previous_A[:,i:i+1].T)
      c=l2_lambda/batch_size
      xw=learn_rate*self.derivative_w
      self.weights -=xw- learn_rate *c*self.weights
      xb=learn_rate * self.derivative_b
      self.bias -=xb-c*self.bias
    return previous_derivative_A

    # rmsprop algorithm for gradient descent
  def rmsprop(self, learn_rate,t,l2_lambda=0,batch_size =32, mrate = 0.9):
    gws=np.square(self.derivative_w)
    gbs = np.square(self.derivative_b)
    nmrate=1-mrate
    self.pv_weight = mrate * self.pv_weight + nmrate * gws
    self.pv_bias = mrate * self.pv_bias + nmrate *gbs
    self.pv_bias[self.pv_bias<0] = 1e-9
    for i in self.pv_weight:
      i[i<0]=1e-9
    a= (learn_rate * l2_lambda / batch_size)
    self.weights=self.weights-a * self.weights
    self.bias= self.bias -a * self.bias
    b=np.sqrt(self.pv_bias+(1e-8))
    c= learn_rate/b
    self.weights = self.weights - c*self.derivative_w
    self.bias = self.bias - c*self.derivative_b
#gradient_descent algorithm
#collection of partial derivatives
  def gradient_descent(self, learn_rate,l2_lambda =0,batch_size =32,t=0):
    c=l2_lambda/batch_size
    self.weights = self.weights - learn_rate * self.derivative_w-learn_rate*c*self.weights
    self.bias = self.bias - learn_rate * self.derivative_b-c*self.bias

 #momentum gradient descent
  def momentum_gradient_descent(self, learn_rate,t,l2_lambda=0,batch_size =32, mrate=0.9):
    c=l2_lambda/batch_size
    self.pm_weight= mrate*self.pm_weight + learn_rate * self.derivative_w+c*self.weights
    self.pm_bias= mrate * self.pm_bias+learn_rate*self.derivative_b+c*self.bias
    self.weights-=self.pm_weight
    self.bias -=self.pm_bias

#nesterov algorithm
  def nesterov(self,learn_rate,mrate = 0.9,l2_lambda =0,batch_size =32,t = 0):
    self.pv_weight = mrate * self.pv_weight + learn_rate * self.derivative_w
    self.weights-=self.pv_weight
    self.bl = self.bias - mrate * self.pv_bias
    self.pv_bias *= mrate

#adam algorithm
  def adam(self,learn_rate , beta1 = 0.9, beta2 = 0.999,l2_lambda =0,batch_size =32,t=0):
    nbeta1=1-beta1;
    nbeta2=1-beta2;
    self.pm_weight = beta1 * self.pm_weight + nbeta1*self.derivative_w
    self.pm_bias = beta1 * self.pm_bias + nbeta1*self.derivative_b
    sw=np.square(self.derivative_w)
    sb=np.square(self.derivative_b)
    self.pv_weight = beta2 * self.pv_weight+ nbeta2*sw
    self.pv_bias = beta2 * self.pv_bias + nbeta2*sb
    self.pm_weightH = self.pm_weight/nbeta1
    self.pm_biasH = self.pm_bias/nbeta1
    self.pv_weightH = self.pv_weight/nbeta2
    self.pv_biasH = self.pv_bias/nbeta2
    rw=np.sqrt(self.pv_weightH+(1e-8))
    self.weights = self.weights - learn_rate * np.divide(self.pm_weightH,rw)
    rb=np.sqrt(self.pv_biasH+(1e-8))
    self.bias = self.bias - learn_rate * np.divide(self.pm_biasH,rb)



#nadam algorithm
  def nadam(self,learn_rate ,t, beta1 = 0.9, beta2 = 0.999,l2_lambda =0,batch_size =32):
    nbeta1=1-beta1
    nbeta2=1-beta2
    self.pm_weight=beta1*self.pm_weight+nbeta1*self.derivative_w
    self.pm_bias=beta1*self.pm_bias+nbeta1*self.derivative_b
    sw=np.square(self.derivative_w)
    sb=np.square(self.derivative_b)
    self.pv_weight=beta2*self.pv_weight+nbeta2*sw
    self.pv_bias=beta2*self.pv_bias+nbeta2*sb
    self.pm_weightH = (beta1 * self.pm_weight /nbeta1) + self.derivative_w
    self.pm_biasH = (beta1 * self.pm_bias /nbeta1) + self.derivative_b
    self.pv_weightH = (beta2 * self.pv_weight) / nbeta2
    self.pv_biasH = (beta2 * self.pv_bias) / nbeta2
    sw=np.sqrt(self.pv_weightH+(1e-8))
    aw=np.divide(self.pm_weightH,sw)
    sb= np.sqrt(self.pv_biasH+(1e-8))
    ab= np.divide(self.pm_biasH,sb)
    self.weights -= (learn_rate *aw)
    self.bias -= (learn_rate *ab)

# QUESTION 2 AND QUESTION 3 FORWARD PROPAGAE AND BACKWARD PROPAGATE
#feedforwardneuralnetwork for calculating all the layers over the model and loss functions are calculated
#with input parameters optimizer,activation function weight initialization
#number of epochs and size of each layer ,learnig rate theeta

class FeedForwardNeuralNetwork:

    def __init__(self, layers_size,epochs=5,learning_rate=0.001, l2_lambda = 0,optimizer = 'gradient_descent', activation = 'sigmoid',weight_type = 'random', loss='cross_entropy'):
        self.layers=[]
        self.layers_size = layers_size
        self.epochs = epochs
        self.learning_rate = learn_rate
        self.optimizer = optimizer
        self.activation = activation
        self.weight_type = weight_type
        self.l2_lambda = l2_lambda
        #checking for type of loss function to calculate
        if loss =='mean_squared_error':
            self.losscomputation = self.mean_square
            self.lossBackwardpass = self.mean_square_grad
        elif loss =='cross_entropy':
            self.losscomputation = self.cross_entropy
            self.lossBackwardpass = self.cross_entropy_grad
        else:
            print('loss computation is invalid')
        self.loss=loss

  # addition of  layer to the feedforward neural network
  #calling to singlelayer from here and required parameters are passed by using adding layer function
    def addingLayer(self, idim=None, nof_nodes=1, activation='', weight_type='random'):
        if not self.layers:
            if idim is None:
              print('Invalid number of layers')
        else:
            if idim is None:
              idim = self.layers[-1].outputDimension()
        add_layer = SingleLayer(idim,nof_nodes, activation, optimizer=self.optimizer, weight_type=weight_type)
        self.layers.append(add_layer)

    # mean_square error
    def mean_square(self, Y, A):
        l=np.square(Y - A)
        b_sum=np.sum(l)
        a= Y.shape[1]
        c = 1 / a * b_sum
        return np.squeeze(c)

    # mean_square_grad
    def mean_square_grad(self, Y, A):
        x=Y-A
        dA = -2 * x
        return dA


    # cross_entropy
    def cross_entropy(self, Y, A):
        a = Y.shape[1]
        b=np.sum(Y*np.log(A))
        c = -(1/a)*b
        return np.squeeze(c)


    # cross_entropy_grad
    def cross_entropy_grad(self, Y, A):
        dA = A-Y
        return dA


    # to get loss from predicted values and true values for given input data
    def cost(self, Y, A):
        return self.losscomputation(Y, A)


    # Forwarding X through all layers present in the model
    def forward_propagate(self, X):
        result = np.copy(X)
        for each_layer in self.layers:
            result = each_layer.forward_propagate(result)
        return result


    # Backward pass Y and A in reverse direction
    def backward_propagate(self, Y, A):
        derivative_A = self.lossBackwardpass(Y, A)
        if self.optimizer != 'stochastic_gradient_descent':
            for each_layer in reversed(self.layers):
                derivative_A = each_layer.backward_propagate(derivative_A)
        elif self.optimizer =='stochastic_gradient_descent':
            for each_layer in reversed(self.layers):
                derivative_A = each_layer.stochastic_gradient_descent(derivative_A,learn_rate = self.learn_rate)


    # Update the weights and calculate gradient descent of all the layers
    def update_Weight(self, learning_rate=0.01,l2_lambda =0,batch_size=32,t=0):
        for each_layer in self.layers:
            each_layer.optimizer(learning_rate,l2_lambda = l2_lambda,batch_size = batch_size,t=0)


    # Training function to train  data for validation and test data from mnist data
    def fit(self,x_train,y_train,x_test,y_test,batch_size = 32):

        from sklearn.model_selection import train_test_split
    #assigning data values from validation and test data by splitting
        x,x_value,y,y_value = train_test_split(x_train,y_train,train_size = 0.9, test_size = 0.1, random_state=10)

        if self.activation=='relu':
          self.weight_type = 'xavier'
    #adding the layers to the model  with activation functions by calling adding layer
        l=len(self.layers_size)
        for k in range(1,l-1):
          self.addingLayer(idim=self.layers_size[k-1], nof_nodes=self.layers_size[k], activation=self.activation, weight_type = self.weight_type)


    # softmax activation function for the last l layer output layer
        self.addingLayer(idim=self.layers_size[-2], nof_nodes=self.layers_size[-1], activation='softmax', weight_type = self.weight_type)

    #one hot encoder
        i=len(y)
        j=len(set(y))
        y_encoder = np.zeros([j,i])
        for k in range(y_encoder.shape[1]):
          y_encoder[y[k]][k] = 1
  #iterating in epochs
  #Training  the data through epochs
        for i in range(self.epochs):

  #avoid gradient vanishing and decreasing the learn_rate if the condition is met for activation function relu
          if self.activation =='relu':
            if self.optimizer == 'momentum'  or self.optimizer == 'nag'or self.optimizer == 'rmsprop':
              d=self.learning_rate/15
              self.learning_rate=d
          #Training the data for each batch size in chunks
          for k in range(0,x.shape[0],batch_size):
            kbatch=k+batch_size
            xbatch = x[k:kbatch]
            ybatch = y[k:kbatch]
            y_encoderbatch = y_encoder[:,k:kbatch]
            xbatch = xbatch.reshape(xbatch.shape[0],xbatch.shape[1]*xbatch.shape[2]).T
            mi=np.min(xbatch)
            mx=np.max(xbatch)
            xbatch = xbatch-mi/mx-mi
            #feed forward neural network and perform backward propagation for the ruuning of network
            if self.optimizer == 'sgd':
              A = self.forward_propagate(xbatch)
              self.backward_propagate(y_encoderbatch,A)
            elif self.optimizer != 'sgd':
              A = self.forward_propagate(xbatch)
              self.backward_propagate(y_encoderbatch,A)
              self.update_Weight(learn_rate=self.learn_rate,l2_lambda = self.l2_lambda,batch_size=batch_size,t= i+1)
          #accuracy and predicted labels for validation data and test data
          #predicting the loss
          validation_loss,validation_acc,_=self.predict(x_value,y_value)
          loss,accuracy,y_pred= self.predict(x_test,y_test)

          #display  loss and accuracy for validation data and test data
          print("After ",i+1,"iterations:")
          print("validation loss;",validation_loss,"validation accuracy:",validation_acc)
          print("test_loss:",loss,"test accuracy:",accuracy)

          #for each epoch entry accuracy and loss in wandb panel by using log function
          wandb.log({"val_loss":validation_loss,"val_accuracy":validation_acc,"loss":loss,"accuracy":accuracy,"epoch":i})

        return y_pred
        #return the probabilistic distributions of each class y-pred


    #predicting loss and accuracy
    def predict(self,x,y):
        i=x.shape[0]
        j=x.shape[1]*x.shape[2]
        A = x.reshape(i,j).T
        a=len(set(y))
        b=len(y)
        y_encoder = np.zeros([a,b])
        for k in range(y_encoder.shape[1]):
          y_encoder[y[k]][k] = 1

        for each_layer in self.layers:
            A = each_layer.predict(A)
        x=-(y_encoder * np.log(A))
        cross_entropy = x.mean() * y_encoder.shape[0]
        y_pred = np.argmax(A,axis = 0)
        accuracy = (y==y_pred).mean()

        return cross_entropy,accuracy,A


 np.random.seed(1)
    model = FeedForwardNeuralNetwork(layers_size = [784]+[arguments.hidden_size]*arguments.num_layers+[10],epochs = arguments["epochs"],learning_rate = arguments.learning_rate,l2_lambda = arguments.weight_decay,loss='cross_entropy',activation = arguments.activation, optimizer = arguments.optimizer, weight_type=arguments.weight_init)
    y_pred = model.fit(x_train,y_train,x_test,y_test,batch_size=config.batch_size)