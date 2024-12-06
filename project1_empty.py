import numpy as np
from matplotlib import pyplot as plt
import sys, math
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

    #initialize neuron with activation type, number of inputs, learning rate, and possibly with set weights
    def __init__(self,activation, input_num, lr, weights=None):

        self.activation = activation
        self.input_num = input_num
        self.lr = lr

        # if set weights are passed
        if weights is not None:
            self.weights = weights
        # weights must be randomized
        else:
            self.weights = np.random.rand(input_num+1)

        self.input = None
        self.output = 0
        self.net = 0
        self.wdeltavector = None
        self.gradient = []
        #print('Neuron constructor')
        
    #This method returns the activation of the net
    def activate(self,net):

        x = net

        # linear
        if self.activation == 0:
            return x
        # logistic
        elif self.activation == 1:
            return 1/(1+math.exp(-x))
        else:
            print('Activation function not supported.')
        #print('activate')
        
    #Calculate the output of the neuron should save the input and output for back-propagation.
    #if input is a multi demsionalarray then we can loop through the array and get the summation
    def calculate(self,input):

        # creating a variable to store the input of the neuron that is later called to the activation function
        self.input = input
        net = 0

        # is input coming in as an array, if so we need a summation function as sum(xi+wi)+b
        for x in range(len(self.weights)-1):
            net += (input[x] * self.weights[x])

        # adding the bias to net with input for it assumed 1
        self.net = net + self.weights[len(self.weights)-1]
        #print("the net of the nueron is ",self.net)

        # output of neuron
        self.output = self.activate(self.net)
        #print("output from FF Nuron ",self.output)

        return self.output  
        #print('calculate')

    #This method returns the derivative of the activation function with respect to the net   
    def activationderivative(self):

        # linear
        if self.activation == 0:
            return 1
        # logistic
        elif self.activation == 1:
            #print("this is neuron output ",self.output)
            #print("output of the activation derivative",self.output * (1 - self.output))
            return (self.output * (1 - self.output))
        else:
            print('Activation function derivative not supported.')
        #print('activationderivative')
    
    #This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer
    def calcpartialderivative(self, wtimesdelta):

        # weights without the bias
        weights = self.weights[0:len(self.weights)-1]

        # calculation of wdeltavector
        self.wdeltavector = np.array(weights) * self.activationderivative()
        self.wdeltavector *= wtimesdelta

        # calculation of partial derivative for each weight (gradient)
        for i in range(len(self.input)):
            self.gradient.insert(i,wtimesdelta * self.activationderivative() * self.input[i])

        # adding and calculating the partial derivative of the bias weight
        self.gradient.append(wtimesdelta * self.activationderivative() * 1)

        #print("weights",weights)
        #print("wtimesdelta:",wtimesdelta)
        #print("this is self.wdeltavector ",self.wdeltavector)

        return self.wdeltavector
        #print('calcpartialderivative')
    
    #Simply update the weights using the partial derivatives and the learning weight
    def updateweight(self):

        for x in range(len(self.weights)):
            self.weights[x] = self.weights[x] - self.lr * self.gradient[x]
        #print('updateweight', self.weights)

        
#A fully connected layer    
# need to add the connection to the other neurons    
class FullyConnected:
    #initialize with the number of neurons in the layer, their activation,the input size, the leraning rate and a 2d matrix of weights (or else initilize randomly)
    def __init__(self,numOfNeurons, activation, input_num, lr, weights=None):

        self.numOfNeurons = numOfNeurons
        self.activation = activation
        self.input_num = input_num
        self.lr = lr
        self.weights = weights

        self.layer = []
        self.output = []
        self.sumwdelta = []

        # initializing a layer of neurons
        for i in range(self.numOfNeurons):
            # if set weights are passed
            if self.weights is not None:
                self.layer.append(Neuron(self.activation, self.input_num, self.lr, self.weights[i]))
            # weights must be randomized
            else:
                self.layer.append(Neuron(self.activation, self.input_num, self.lr))
        #print('Fully connected constructor')
        
        
    #calculate the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calcualte() method)
    #input should be coming in as a vector 
    def calculate(self, input):

        #print("in Fully connect calculate")
        # create array and add the results of calculate to the array
        for i in range(self.numOfNeurons):
            self.output.insert(i,(self.layer[i].calculate(input)))

        #print ("output from calc in FC Layer",self.output)
        return (self.output)
        #print('calculate in Fully connected')
    
    #given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the correct value), sum up its ownw*delta, and then update the wieghts (using the updateweight() method). I should return the sum of w*delta.          
    def calcwdeltas(self, wtimesdelta):
        #print("this is wtimedelta ",wtimesdelta)

        # testing for example
        #print("neuron 0 output:", self.layer[0].output)
        #print("neuron 1 output:", self.layer[1].output)

        presumwdelta = []

        # get onew*delta through calcpartialderivative()
        for i in range(self.numOfNeurons):
            ownwtimesdelta =(self.layer[i].calcpartialderivative(wtimesdelta[i]))
            #print("this is owntimedelta ",ownwtimesdelta)     #*wtimesdelta))
            presumwdelta.insert(i,ownwtimesdelta)

            #update weights
            self.layer[i].updateweight()

        # sum up its ownw*delta
        self.sumwdelta = np.sum(presumwdelta, axis=0)

        #print("This is the array from calcwdeltas",self.sumwdelta)
        return self.sumwdelta
        print('calcwdeltas') 
           
        
#An entire neural network        
class NeuralNetwork:
    #initialize with the number of layers, number of neurons in each layer (vector), input size, activation (a vector, one for each layer), the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)
    def __init__(self,numOfLayers,numOfNeurons, inputSize, activation, loss, lr, weights=None):
        
        self.numOfLayers = numOfLayers
        self.numOfNeurons = numOfNeurons
        self.input_num = inputSize
        self.activation = activation
        self.loss = loss

        self.network = []
        self.eTotal=0
        self.etotaldirv = []
        self.output =[]

        # if set weights are passed
        if weights is not None:
            for i in range(numOfLayers):
                self.network.insert(i,FullyConnected(numOfNeurons[i], activation[i], self.input_num, lr, weights[i]))
                self.input_num = numOfNeurons[i]
        # weights must be randomized
        else:
            for i in range(numOfLayers):
                self.network.insert(i,FullyConnected(numOfNeurons[i], activation[i], self.input_num, lr))
                self.input_num = numOfNeurons[i]
        #print('constructor complete')
    
    #Given an input, calculate the output (using the layers calculate() method)
    def calculate(self,input):

        # feed forward
        for i in range(self.numOfLayers):
            ninput = self.network[i].calculate(input)
            input = ninput
        self.output =ninput

        # print("This is the out put from the NN Class",self.output)
        return(self.output)
       #print('NN calculate, creating the fully connected layer')
        
    #Given a predicted output and ground truth output simply return the loss (depending on the loss function)
    def calculateloss(self,yp,y):
        # set to zero
        self.eTotal = 0

        # sum of squares
        if self.loss == 0:
            for i in range(len(y)):
                self.eTotal +=(.5)*((y[i]-yp[i])**2)
        # binary cross entropy
        if self.loss == 1: #binary cross entropy
            sum=0
            for i in range(len(y)-1):
                sum += -(y[i] * np.log(yp[i]) + (1-y[i])*np.log(1-yp[i]))
            self.eTotal = -(sum)/len(y)
        # print("this is the etotal",self.eTotal)
        return self.eTotal
        #print('calculate')
    
    #Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)        
    def lossderiv(self,yp,y):

        # sum of squares
        if self.loss == 0:
            etotaldirv = (-1)*(y-yp)
         # binary cross entropy
        if self.loss == 1:
            if (yp == 0):
                etotaldirv = -(y) + ((1 - y) / (1 - yp))
            elif (yp == 1):
                etotaldirv = -(y/yp) + ((1 - y) / (1))
            else:
                etotaldirv = -(y/yp)+((1-y)/(1-yp))

        return etotaldirv
        #print('lossderiv')
    
    #Given a single input and desired output preform one step of backpropagation (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values         
    def train(self,x,y):

        #print("starting the NN Calculate")
        output = self.calculate(x)

        #print("This is the output of FF NN",self.output)
        #print("Feed forward has completed")

        # calculate total loss
        loss = self.calculateloss(output[0:len(y)],y)

        # get etotalloss derivative
        for x in range(len(y)):
            self.etotaldirv.insert(x,self.lossderiv(self.output[x],y[x]))
        wdeltas = self.etotaldirv

        # backprogation to update weights
        for x in range(len(self.network)-1, -1, -1):
            wdeltas = self.network[x].calcwdeltas(wdeltas)
            
        return loss
        print('train')

if __name__=="__main__":
    if (len(sys.argv)<2):

        # create graphs using example
        x = np.array([0.05, 0.1])
        yp = np.array([0.01, 0.99])
        lr = np.linspace(0.1, 0.7, 14)
        lossPerEpoch = []

        for l in lr:

            # reset w every time and network
            w = np.array([[[.15, .2, .35], [.25, .3, .35]], [[.4, .45, .6], [.5, .55, .6]]])
            network = NeuralNetwork(2,[2,2],2,[1,1],0,l,w)

            lossPerEpochForEachlr = []

            for i in range(250):
                lossPerEpochForEachlr.append(network.train(x, yp))

            lossPerEpoch.append(lossPerEpochForEachlr)

        # plotting loss per epoch pe learning rate
        for i in range(len(lossPerEpoch)):
            plt.plot(lossPerEpoch[i], label=f'Learning Rate: {lr[i]:.2f}')

        plt.xlabel('Number of Epochs')
        plt.ylabel('Total Loss')
        plt.legend()
        plt.title('Total Loss vs Number of Epochs')
        plt.show()
        
    elif (sys.argv[2]=='example'):

        print('run example from class (single step)')
        w=np.array([[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]])
        x=np.array([0.05,0.1])
        yp=np.array([0.01,0.99])
        lr= float(sys.argv[1])

        #setting up the neural network
        network=NeuralNetwork(2,[2,2], 2,[1,1],0,lr,w)

        #training neural network
        network.train(x,yp)

        print(f"    Output: {[round(l, 3) for l in network.calculate(x)[0:len(yp)]]}")
        print(f"    Total Loss: {network.calculateloss(network.calculate(x)[0:len(yp)],yp)}")
        print(f"Updated Weights:")
        for i in range(len(network.network)):
            print(f"    Layer {i + 1}:")
            for j in range(len(network.network[i].layer)):
                print(f"        Neuron {j + 1}: {network.network[i].layer[j].weights}")



    elif(sys.argv[2]=='and'):
        print('learn and')

        lr = float(sys.argv[1])
        network = NeuralNetwork(1, [1], 2, [0], 0, lr, None)

        for i in range(1500):
            network.train([0, 0], np.array([0]))
            network.train([1, 0], np.array([0]))
            network.train([0, 1], np.array([0]))
            network.train([1, 1], np.array([1]))

        print('Outputs - Converged:')
        print(f"    [0,0]: {round(network.calculate([0, 0])[0])}")
        print(f"    [1,0]: {round(network.calculate([1, 0])[0])}")
        print(f"    [0,1]: {round(network.calculate([0, 1])[0])}")
        print(f"    [1,1]: {round(network.calculate([1, 1])[0])}")

        for x in range(len(network.network)):
            for y in range(len(network.network[x].layer)):
                print(f"    Layer {x + 1} Neuron {y + 1} weights: {[round(l, 2) for l in network.network[x].layer[y].weights]}")

    elif(sys.argv[2]=='xor'):
        print('learn xor')

        lr = float(sys.argv[1])

        #single perceptron
        network = NeuralNetwork(1, [1], 2, [1], 1, lr, None)

        for i in range(1500):
            network.train([0, 0], np.array([0]))
            network.train([1, 0], np.array([1]))
            network.train([0, 1], np.array([1]))
            network.train([1, 1], np.array([1]))

        print('Outputs - Converged: Single Perceptron')
        print(f"    [0,0]: {round(network.calculate([0, 0])[0])}")
        print(f"    [1,0]: {round(network.calculate([1, 0])[0])}")
        print(f"    [0,1]: {round(network.calculate([0, 1])[0])}")
        print(f"    [1,1]: {round(network.calculate([1, 1])[0])}")

        for x in range(len(network.network)):
            for y in range(len(network.network[x].layer)):
                print(f"    Layer {x + 1} Neuron {y + 1} weights: {[round(l, 2) for l in network.network[x].layer[y].weights]}")


        # one hidden layer
        network1 = NeuralNetwork(2, [1,1], 2, [0,1], 1, lr, None)

        for i in range(150):
            network1.train([0, 0], np.array([0]))
            network1.train([1, 0], np.array([1]))
            network1.train([0, 1], np.array([1]))
            network1.train([1, 1], np.array([1]))

        print('Outputs - Converged: One Hidden Layer')
        print(f"    [0,0]: {round(network1.calculate([0, 0])[0])}")
        print(f"    [1,0]: {round(network1.calculate([1, 0])[0])}")
        print(f"    [0,1]: {round(network1.calculate([0, 1])[0])}")
        print(f"    [1,1]: {round(network1.calculate([1, 1])[0])}")

        for x in range(len(network1.network)):
            for y in range(len(network1.network[x].layer)):
                print(f"    Layer {x + 1} Neuron {y + 1} weights: {[round(l, 2) for l in network1.network[x].layer[y].weights]}")
