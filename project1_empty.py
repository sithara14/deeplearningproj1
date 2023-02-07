import numpy as np
import sys, random, math

"""
For this entire file there are a few constants:
activation:
0 - linear
1 - logistic (only one supported)
loss:
0 - sum of square errors
1 - binary cross entropy
"""


# A class which represents a single neuron
class Neuron:
    # initilize neuron with activation type, number of inputs, learning rate, and possibly with set weights
    # weights if null have to be random from 0-1
    def __init__(self, activation, input_num, lr, weights=None):
        self.activation = activation
        self.input_num = input_num
        self.lr = lr
        self.weights = weights
        # setting up the output and input history
        self.wdeltavector = []
        self.input = []
        self.output = 0
        self.partial_deriv = 0
        self.net = 0  # we are treating the net as the input to the activation function

        print('constructor')

        # This method returns the activation of the net

    def activate(self, net):
        x = net
        if self.activation == 0:
            return x
        elif self.activation == 1:
            return 1 / (1 + math.exp(-x))

        print('activate')

        # Calculate the output of the neuron should save the input and output for back-propagation.

    # is input coming in as an array, if so we need a summation function as sum(xi+wi)+b
    # if input is a multi demsionalarray then we can loop through the array and get the summation
    def calculate(self, input):
        # if weights are none fill with random numbers
        if self.weights == None:
            self.weights = np.random.rand(1,
                                          self.input_num + 1)  # +1  for the bias set to one in requierments of the project
        i = input  # store input in the instance of the nueron
        i.append(1)
        self.input = i

        net = 0  # creating a variable to store the input of the nueron that is later called to the activation function
        for x in range(self.input_num + 1):
            net += (i[x] * self.weights[x])  # doing the simga i=0 to n wi*
        self.net = net
        self.output = self.activate(self, net)

        return self.output  # not sure if we need to return anything here
        print('calculate')

    # This method returns the derivative of the activation function with respect to the net
    def activationderivative(self):
        if self.activation == 0:
            return 1
        elif self.activate == 1:
            return (self.net * (1 - self.net))
        print('activationderivative')

        # This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer

    def calcpartialderivative(self, wtimesdelta):
        weights = self.weights[0:self.input_num]
        self.wdeltavector = [item * wtimesdelta * self.activationderivative(self) for item in weights]
        return self.wdeltavector
        print('calcpartialderivative')

        # Simply update the weights using the partial derivatives and the learning weight

    def updateweight(self):
        lastinput = self.input_history[len(self.input_history) - 1, 0:self.input_num]
        gradient = np.multiply(self.wdeltavector, lastinput)

        for x in range(self.weights - 1):
            self.weights[x] = self.weights[x] - self.lr * gradient[x]
        print('updateweight')


# A fully connected layer
# need to add the connection to the other neurons
class FullyConnected:
    # initialize with the number of neurons in the layer, their activation,the input size, the leraning rate and a 2d matrix of weights (or else initilize randomly)
    def __init__(self, numOfNeurons, activation, input_num, lr, weights=None):
        self.numOfNeurons = numOfNeurons
        self.activation = activation
        self.input_num = input_num
        self.lr = lr
        self.weights = weights
        self.layer = []

        # initializing layer of neuron
        for i in range(self.numOfNeurons):
            if self.weights is not None:
                self.layer.append(Neuron(self.activation, self.input_num, self.lr, self.weights[i]))
            else:
                self.layer.append(Neuron(self.activation, self.input_num, self.lr))

        print('constructor')

        # calcualte the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calcualte() method)

    # input should be coming in as a vector
    def calculate(self, input):
        # create array and add the results of calculate to the array
        output = 0
        for i in input:
            self.layer.calculate(input)
        print('calculate')

        # given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the correct value), sum up its ownw*delta, and then update the wieghts (using the updateweight() method). I should return the sum of w*delta.

    def calcwdeltas(self, wtimesdelta):
        print('calcwdeltas')

    # An entire neural network


class NeuralNetwork:
    # initialize with the number of layers, number of neurons in each layer (vector), input size, activation (a vector, one for each layer), the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)
    def __init__(self, numOfLayers, numOfNeurons, inputSize, activation, loss, lr, weights=None):

        self.numOfLayers = numOfLayers
        self.numOfNeurons = numOfNeurons
        self.input_num = inputSize
        self.activation = activation
        self.loss = loss
        self.network = []
        self.output = []
        self.eTotal = 0
        for i in range(numOfLayers):
            network.append(FullyConnected(numOfNeurons[i], activation[i], inputSize, lr, weights[i]))

        print('constructor complete')

        # Given an input, calculate the output (using the layers calculate() method)

    def calculate(self, input):

        for i in range(
                self.numOfLayers - 1):  # I did the this so we can save the output of the calculate the total loss for the Etotal and backprop
            if i == self.numOfLayers - 2 or i == self.numOfLayers - 1:
                self.output.append(network.calculate(input))
            else:
                network[i].calculate(input)
        print('constructor')

    # Given a predicted output and ground truth output simply return the loss (depending on the loss function)
    def calculateloss(self, yp, y):
        return (yp - p)
        print('calculate')

    # Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)
    def lossderiv(self, yp, y):
        return self.calculateloss(yp, y) * FullyConnected.Neuron.activationderivative(y)

        print('lossderiv')

    # Given a single input and desired output preform one step of backpropagation (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values
    def train(self, x, y):
        output = self.calculate(x)

        # finding the E total of the network
        for i in range(output):
            self.eTotal += self.lossderiv(y, self.output)
        ######################
        # stopping here my brain hurts
        print('train')


if __name__ == "__main__":
    user_input = input()
    if (len(sys.argv) < 2):
        print('a good place to test different parts of your code')

    elif (sys.argv[1] == 'example'):
        print('run example from class (single step)')
        w = np.array([[[.15, .2, .35], [.25, .3, .35]], [[.4, .45, .6], [.5, .55, .6]]])
        x = np.array([0.05, 0.1])
        yp = np.array([0.01, 0.99])

        # setting up the neural network
        network = NeuralNetwork(2, [2, 2], 2, [1, 1], 0, .5, w)
        network.train(x, yp)




    elif (sys.argv[1] == 'and'):
        print('learn and')

    elif (sys.argv[1] == 'xor'):
        print('learn xor')
