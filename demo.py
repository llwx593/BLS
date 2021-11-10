import numpy as np
import scipy.io as scio
from base_bls import BaseBls
import time

def show_accuracy(predictLabel, Label): 
    count = 0
    label_1 = np.zeros(Label.shape[0])
    predlabel = []
    label_1 = Label.argmax(axis=1)
    predlabel = predictLabel.argmax(axis=1)
    for j in list(range(Label.shape[0])):
        if label_1[j] == predlabel[j]:
            count += 1
    return (round(count/len(Label),5))

dataFile = './mnist.mat'
data = scio.loadmat(dataFile)
traindata = np.double(data['train_x']/255)
trainlabel = np.double(data['train_y'])
testdata = np.double(data['test_x']/255)
testlabel = np.double(data['test_y'])

N1 = 10
N2 = 10
N3 = 500
s = 0.8
C = 2**-30

print("------BLS_BASE------")
bls = BaseBls(N1, N2, N3, s, C)
start_time = time.time()
train_out = bls.forward(traindata, trainlabel)
end_time = time.time()
train_time = end_time - start_time
train_acc = show_accuracy(train_out, trainlabel)
print('Training accurate is' ,train_acc*100,'%')
print('Training time is ',train_time,'s')

bls.eval()
start_time = time.time()
predict_out = bls.forward(testdata)
end_time = time.time()
test_time = end_time - start_time
test_acc = show_accuracy(predict_out, testlabel)
print('Testing accurate is' ,test_acc * 100,'%')
print('Testing time is ',test_time,'s')
