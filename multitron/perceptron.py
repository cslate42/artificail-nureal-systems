from random import choice
# from numpy import array, dot, random
import numpy as np

def unitStep(x):
    """ 1 if > 0 else 0 """
    return 1 if x > 0 else 0

def create( n = 1, sizeOfData = 1 ):
    """
    create 'n' perceptrons and return array
    """
    perceptrons = []
    for i in range(n):
        perceptrons.append( perceptron(sizeOfData) )
    return perceptrons

class perceptron(object):
    """
    Learning algorithm
    supply me with training DATA
    update weights
    "Learn" weights needed for correct answer
    """

    learningRate = 0.2
    # TODO remove??? dont think needed since multitron runs?
    interations = 100

    def __init__(self, sizeOfData):
        """
        create variables
        set
        """
        self.weights = np.random.rand( sizeOfData )#np.random.rand(weights)

        return

    def train(self, inputRow, isInputTrue):
        """
        Update weights based on training data
        inputRow - the training data
        isInputTrue - boolean is training data correct
        """
        # multiply trainingData subarray by weights then sum answer
        result = np.dot(self.weights, inputRow)
        # update error?
        error = isInputTrue - unitStep(result)
        # update weights
        self.weights += self.learningRate * error * inputRow

        return

    def initializeWeights(self, trainingData):
        """
        TODO delete??? don't use
        """
        for i in range(iterations):
            #choose random row from training data
            inputRow, isTrue = random.choice(trainingData)

            # multiply trainingData subarray by weights then sum answer
            result = np.dot(self.weights, inputRow)
            # update error?
            error = isTrue - unitStep(result)
            # update weights
            self.weights += self.learningRate * error * inputRow
        return

    def isDataValid(self, data):
        """
        given a row of data return if rates
        dotted with row is > 0 for true
        returns 1 or 0
        """
        return unitStep( np.dot(data, self.weights) )

    def outputResults(data):
        """
        TODO finish?
        """
        # output learned results
        # inputRow, ignored
        # for inputRow, _ in data:
        #         result = dot(inputRow, weights)
        #         print("{}: {} -> {}".format(inputRow, result, unitStep(result) ))
        return
