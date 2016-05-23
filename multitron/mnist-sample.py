import random, pickle
with open('./littlemnist.pkl', 'rb') as f:
    (trainX, trainY), (validX, validY), (testX, testY) = pickle.load(f,encoding= 'latin1')
print('loaded data')
train = [(x,y)for x,y in zip (trainX, trainY)]
valid = [(x,y)for x,y in zip (validX, validY)]
test =  [(x,y)for x,y in zip (testX, testY)]


print("test[0]:{} \nlen(test[0])={}\nlen(test[0][])={}\n num={}".format(test[0], len(test[0]), len(test[0][0]), test[0][1]));
