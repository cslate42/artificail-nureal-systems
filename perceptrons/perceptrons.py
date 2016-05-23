from random import choice
from numpy import array, dot, random

def unitStep(x):
    return 1 if x > 0 else 0

# N bit or
trainingData = [
    # 1 bit
    # (array([0]), 0),
    # (array([1]), 1),

    # 2 bit
    # (array([0,0]), 0),
    # (array([0,1]), 1),

    # 3 bit
    # (array([0,0,0]), 0),
    # (array([0,1,1]), 1),
    # (array([1,0,1]), 1),
    # (array([1,1,1]), 1),

    # 4 bit
    (array([0,0,0,0]), 0),
    (array([0,0,0,1]), 1),
    (array([0,0,1,0]), 1),
    (array([0,1,0,0]), 1),
    (array([1,0,0,0]), 1),
    (array([1,1,0,0]), 1),
    (array([1,0,1,0]), 1),
]
weights = random.rand(4)
learningRate = 0.2
iterations = 100

def initializeWeights():
    global trainingData, weights, learningRate, iterations
    for i in range(iterations):
        #choose random row from training data
        inputRow, isTrue = choice(trainingData)

        # multiply trainingData subarray by weights then sum answer
        result = dot(weights, inputRow)
        # update error?
        error = isTrue - unitStep(result)
        # update weights
        weights += learningRate * error * inputRow

def outputResults(data):
    # output learned results
    # inputRow, ignored
    for inputRow, _ in data:
            result = dot(inputRow, weights)
            print("{}: {} -> {}".format(inputRow, result, unitStep(result) ))
