import random, pickle
import numpy as np
import matplotlib
import perceptron

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
    numOfCharacters = 10
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
        """
        Load part of the MNIST data
        3 parts
        training data  10,000 elements
            2D array (
                array( 28x28 bitmap ), correctNumber[0-9]
            )
        validation data 1,000 elements
            2D array (
                array( 28x28 bitmap ), correctNumber[0-9]
            )
        testing data 1,000 elements
            2D array (
                array( 28x28 bitmap ), correctNumber[0-9]
            )
        """
        with open('./littlemnist.pkl', 'rb') as f:
            (trainX, trainY), (validX, validY), (testX, testY) = pickle.load(f,encoding= 'latin1')
        print('loaded data')
        self.trainingData = [(x,y)for x,y in zip (trainX, trainY)]
        self.validData = [(x,y)for x,y in zip (validX, validY)]
        self.testingData =  [(x,y)for x,y in zip (testX, testY)]

        # print( "TRAINING", len(self.trainingData), "VALID", len(self.validData), "TESTING", len(self.testingData), self.validData[0] )
        # exit(0);

        sizeOfData = len( self.trainingData[0][0] )
        # array of perceptrons for each number
        self.perceptrons = perceptron.create( self.numOfCharacters, sizeOfData )
        # print(self.trainingData[0][0].size)
        self.weights = np.random.rand( self.numOfCharacters )
        return

    def train(self):
        """
        Use training data to update weights
        """
        numOfTrainingData = len( self.trainingData )
        numOfValidData = len( self.validData )

        # loop through all training data
        for i in range(numOfTrainingData):
            # train each character with training data row
            for j in range( self.numOfCharacters ):
                trainingData = self.trainingData[i][0]
                isCorrect = 1 if j == self.trainingData[i][1] else 0
                self.perceptrons[j].train( trainingData, isCorrect )

            # every divisable peice of training data
            # redo weights with valid data
            if( (i % numOfValidData) == 0 ):
                for j in range( len(self.validData) ):
                    trainingData = self.validData[j][0]
                    num = self.validData[j][1]
                    self.perceptrons[ num ].train( trainingData, 1 )

        return

    def validate(self):
        """
        Rerun training with known values to update weights
        ??? should be run every inbetween 'x' peices of training data
        """
        return

    def test(self):
        """
        go through testing data and show which ones output correct weights
        """
        count = 0;
        for i in range( len(self.testingData) ):
            if( self.testingData[i][1] == 3 ):
                count += 1
                isValid = self.perceptrons[3].isDataValid(self.testingData[i][0])
                print("\nRow[{}]: is3={} is8={}".format(count, isValid
                    , self.perceptrons[8].isDataValid(self.testingData[i][0])
                ) )
        return

    def printTrainingDataInfo(self):
        """
        TODO delete? just debug... not useful right now
        """
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
