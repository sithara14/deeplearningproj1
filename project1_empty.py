
import numpy as np
import sys,random, math
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
    #initilize neuron with activation type, number of inputs, learning rate, and possibly with set weights
    #weights if null have to be random from 0-1
    def __init__(self,activation, input_num, lr, weights=None):
        self.activation = activation
        self.input_num = input_num
        self.lr = lr
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.rand(input_num+1)
        #setting up the output and input history
        self.wdeltavector=[]
        self.input = None
        self.output = 0
        self.gradient = []
        self.net = 0  #we are treating the net as the input to the activation function
        self.wdeltavector = None

        
        #print('Neuron constructor')
        
    #This method returns the activation of the net
    def activate(self,net):
        x = net
        if self.activation == 0 : #linear
            return x
        elif self.activation == 1 : #logistic
            return 1/(1+math.exp(-x))

        #print('activate')
        
    #Calculate the output of the neuron should save the input and output for back-propagation. 
    # is input coming in as an array, if so we need a summation function as sum(xi+wi)+b  
    #if input is a multi demsionalarray then we can loop through the array and get the summation
    def calculate(self,input):
        self.input = input
        net = 0                                 # creating a variable to store the input of the nueron that is later called to the activation function
        for x in range(len(self.weights)-1):
            net += (input[x] * self.weights[x])                     # doing the simga i=0 to n wi*
        self.net = net + self.weights[len(self.weights)-1]#plus 1 for the bias
        #print("the net of the nueron is ",self.net)
        self.output = self.activate(self.net)
        #print("output from FF Nuron ",self.output)

        return self.output  
        print('calculate')

    #This method returns the derivative of the activation function with respect to the net   
    def activationderivative(self):
        if self.activation == 0: #linear
            return 1
        elif self.activation == 1: #logistic
            #print("this is neruone output ",self.output)
            #print("output of activationderv",self.output * (1 - self.output))
            return (self.output * (1 - self.output))
        print('activationderivative')
    
    #This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer
    def calcpartialderivative(self, wtimesdelta):
        weights = self.weights[0:len(self.weights)-1]
        self.wdeltavector = np.array(weights) * self.activationderivative()
        self.wdeltavector *= wtimesdelta

        for i in range(len(self.input)):
            self.gradient.insert(i,wtimesdelta * self.activationderivative() * self.input[i])

        #print("weights",weights)
        #print("wtimesdelta:",wtimesdelta)
        #print("this is self.wdeltavector ",self.wdeltavector)
        return self.wdeltavector
        print('calcpartialderivative')
    
    #Simply update the weights using the partial derivatives and the learning weight
    def updateweight(self):
        for x in range(len(self.weights) - 1):
            self.weights[x] = self.weights[x] - self.lr * self.gradient[x]
        #print('updateweight', self.weights)

        
#A fully connected layer    
# need to add the connection to the other neurons    
class FullyConnected:
    #initialize with the number of neurons in the layer, their activation,the input size, the leraning rate and a 2d matrix of weights (or else initilize randomly)
    def __init__(self,numOfNeurons, activation, input_num, lr, weights=None):
        self.numOfNeurons = numOfNeurons
        self.activation = activation
        self.input_num =input_num
        self.lr = lr
        self.weights = weights
        self.layer = []
        self.output =[]
        self.sumwdelta =[]

        # initializing layer of neuron
        for i in range(self.numOfNeurons):
            if self.weights is not None:
                self.layer.append(Neuron(self.activation, self.input_num, self.lr, self.weights[i]))
            else:
                self.layer.append(Neuron(self.activation, self.input_num, self.lr))


        #print('Fully connected constructor')
        
        
    #calcualte the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calcualte() method)      
    #input should be coming in as a vector 
    def calculate(self, input):
       #create array and add the results of calculate to the array
        #print("in Fully connect calculate")
        
        for i in range(self.numOfNeurons):
            self.output.insert(i,(self.layer[i].calculate(input)))  #[i][]
        #print ("output from calc in FC Layer",self.output)
        return (self.output)
        print('calculatein Fully connected') 
    
    #given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the correct value), sum up its ownw*delta, and then update the wieghts (using the updateweight() method). I should return the sum of w*delta.          
    def calcwdeltas(self, wtimesdelta):
        #print("this is wtimedelta ",wtimesdelta)



        #print("neuron 0 output:", self.layer[0].output)
        #print("neuron 1 output:", self.layer[1].output)

        presumwdelta = []
        for i in range(self.numOfNeurons):
            ownwtimesdelta =(self.layer[i].calcpartialderivative(wtimesdelta[i]))
            #print("this is owntimedelta ",ownwtimesdelta)     #*wtimesdelta))
            presumwdelta.insert(i,ownwtimesdelta)
            self.layer[i].updateweight()


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
        if weights is not None:
            for i in range(numOfLayers):
                self.network.insert(i,FullyConnected(numOfNeurons[i], activation[i], inputSize[i], lr, weights[i]))
        else:
            for i in range(numOfLayers):
                self.network.insert(i,FullyConnected(numOfNeurons[i], activation[i], inputSize[i], lr))

        #print('constructor complete')
    
    #Given an input, calculate the output (using the layers calculate() method)
    def calculate(self,input):

        for i in range(self.numOfLayers):
            ninput = self.network[i].calculate(input)
            input = ninput
        self.output =ninput

        # print("This is the out put from the NN Class",self.output)
        return(self.output)
        print('NN calculate, creating the fully connected layer')
        
    #Given a predicted output and ground truth output simply return the loss (depending on the loss function)
    def calculateloss(self,yp,y):
        if self.loss == 0: #sum of squares
            for i in range(len(yp)):
                self.eTotal +=(.5)*(yp[i]-y[i])**2
            #self.eTotal = (0.5)*(np.sum((np.subtract(yp - y))^2))
        if self.loss == 1: #binary cross entropy
            sum=0
            for i in range(len(y)-1):
                sum += y[i] * np.log(yp[i]) + (1-y[i])*np.log(1-yp[i])
            self.eTotal = -(sum)/len(y)
        #print("this is the etotal",self.eTotal)
        
        #print('calculate')
    
    #Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)        
    def lossderiv(self,yp,y):
        if self.loss == 0: #sum of squares
                etotaldirv = (-1)*(yp-y)
        if self.loss == 1: #binary cross entropy
                etotaldirv = -(y/yp)+((1-y)/(1-yp))

        return etotaldirv
        print('lossderiv')
    
    #Given a single input and desired output preform one step of backpropagation (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values         
    def train(self,x,y):
        #print("starting the NN Calculate")
        self.output = self.calculate(x)
        #print("This is the output of FF NN",self.output)
        #print("Feed forward has completed")

        lossdirv = self.lossderiv(yp, y)

        for x in range(len(yp)):
            self.etotaldirv.insert(x,self.lossderiv(y[x],self.output[x])) 
    
        wdeltas = self.etotaldirv
        for x in range(len(self.network)-1, -1, -1):
            wdeltas = self.network[x].calcwdeltas(wdeltas)
            
        return lossdirv
        print('train')

if __name__=="__main__":
    if (len(sys.argv)<2):
        print('a good place to test different parts of your code')
        
    elif (sys.argv[2]=='example'):
        print('run example from class (single step)')
        w=np.array([[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]])
        x=np.array([0.05,0.1])
        yp=np.array([0.01,0.99])
        #setting up the neural network
        network=NeuralNetwork(2,[2,2],[2,2],[1,1],0,float(sys.argv[1]),w)
        #training neural network
        network.train(x,yp)

        print(f"Updated Weights:")
        for i in range(len(network.network)):
            for j in range(len(network.network[i].layer)):
                print(f"    Layer {i} Neuron {j} weights: {network.network[i].layer[j].weights}")

    elif(sys.argv[2]=='and'):
        
        print('learn and')
        
    elif(sys.argv[2]=='xor'):
        print('learn xor')
