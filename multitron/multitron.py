import random, pickle
import numpy as np
import matplotlib

def stepFunc(x):
    """Typical Step function U(t)"""
    return 1 if x > 0 else 0;

class Multitron(object):
    """
    --Simple Multitron using MNIST Data--
    Basically use array of 10 perceptrons
    to find out weights for bitmap[28x28] of image then decide
    which number it is
    (28x28=784)
    """

    iterations = 200
    learningRate = .2

    def __init__(self):
        """
        Data is list 2D array where
        array[0][0] = input data
            intensity from 0,1
            array dtype=float32
        array[0][1] = correct number
        """
        self.trainingData = []
        self.validData = []
        self.testingData = []
        return

    def loadData(self):
        with open('./littlemnist.pkl', 'rb') as f:
            (trainX, trainY), (validX, validY), (testX, testY) = pickle.load(f,encoding= 'latin1')
        print('loaded data')
        self.trainingData = [(x,y)for x,y in zip (trainX, trainY)]
        self.validData = [(x,y)for x,y in zip (validX, validY)]
        self.testingData =  [(x,y)for x,y in zip (testX, testY)]

        # array of perceptrons for each number
        self.perceptrons = []
        # print(self.trainingData[0][0].size)
        self.weights = np.random.rand( 10 )
        return

    def train(self):
        for i in range(self.iterations):
            #choose random row from training data
            inputRow, isTrue = random.choice(self.trainingData)

            # multiply trainingData subarray by weights then sum answer
            result = np.dot(self.weights, inputRow)
            # update error?
            error = isTrue - unitStep(result)
            # update weights
            self.weights += learningRate * error * inputRow
        return

    def validate(self):
        return

    def test(self):
        return

    def printTrainingDataInfo(self):
        print("test[0]:{} \
            \nlen(test[0])={}\
            \nlen(test[0][])={}\
            \nnum={}".format(
                self.testingData[0],
                len(self.testingData[0]),
                len(self.testingData[0][0]),
                self.testingData[0][1]
            )
        );
        return
