import numpy as np
import scipy.io as scio
from base_bls import BaseBls
import time
from utils import show_mnist_accuracy
from keras.preprocessing.image import ImageDataGenerator

N1 = 10
N2 = 10
N3 = 500
s = 0.8
C = 2**-30

def test_mnist(file_path=None):
    dataFile = file_path
    data = scio.loadmat(dataFile)
    traindata = np.double(data['train_x']/255)
    trainlabel = np.double(data['train_y'])
    testdata = np.double(data['test_x']/255)
    testlabel = np.double(data['test_y'])

    print("------BLS_BASE------")
    bls = BaseBls(N1, N2, N3, s, C)
    start_time = time.time()
    train_out = bls.forward(traindata, trainlabel)
    end_time = time.time()
    train_time = end_time - start_time
    train_acc = show_mnist_accuracy(train_out, trainlabel)
    print('Training accurate is' ,train_acc*100,'%')
    print('Training time is ',train_time,'s')

    bls.eval()
    start_time = time.time()
    predict_out = bls.forward(testdata)
    end_time = time.time()
    test_time = end_time - start_time
    test_acc = show_mnist_accuracy(predict_out, testlabel)
    print('Testing accurate is' ,test_acc * 100,'%')
    print('Testing time is ',test_time,'s')
    return


def test_messidor(dataset="messidor", image_size=512, batch_size=16):
    dataParam={'messidor': [957,243,2,'./datasets/messidor/train','./datasets/messidor/test'],
               'kaggle': [30000,5126,5,'./datasets/kaggle/train','./datasets/kaggle/valid'],
               'DDR':   [9851,2503,5,'./datasets/DDR/train','./datasets/DDR/valid']}
    
    train_num,valid_num,classes,train_dir,test_dir = dataParam[dataset]
    
    train=ImageDataGenerator(horizontal_flip=True,vertical_flip=True,rotation_range=90)          
    valid = ImageDataGenerator()
    train_data=train.flow_from_directory(train_dir,
                                         target_size=(image_size,image_size),
                                         shuffle = True,
                                         batch_size=batch_size)
    valid_data=valid.flow_from_directory(test_dir,
                                         target_size=(image_size,image_size),
                                         shuffle = False,
                                         batch_size=batch_size)

    train_x = np.zeros([train_num, image_size*image_size*3])
    train_y = np.zeros([train_num, classes])

    test_x = np.zeros([valid_num, image_size*image_size*3])
    test_y = np.zeros([valid_num, classes])

    count = 0
    for batch_data in train_data:
        count += 1
        print("now is batch : ", count)
        pre_trainx = batch_data[0]
        pre_trainy = batch_data[1]

        start_index = (count - 1) * batch_size
        if count != 60:
            pre_trainx = np.resize(
                pre_trainx, (batch_size, image_size*image_size*3))
            end_index = count * batch_size
        elif count == 60:
            pre_trainx = np.resize(
                pre_trainx, (13, image_size*image_size*3))
            end_index = start_index + 13
        train_x[start_index : end_index, :] = pre_trainx
        train_y[start_index : end_index, :] = pre_trainy

        if count == 60:
            break
    
    count = 0
    for batch_data in valid_data:
        count += 1
        print("val now is batch : ", count)
        pre_testx = batch_data[0]
        pre_testy = batch_data[1]

        start_index = (count - 1) * batch_size
        if count != 16:
            pre_testx=np.resize(
                pre_testx, (batch_size, image_size*image_size*3))
            end_index = count * batch_size
        elif count == 16:
            pre_testx=np.resize(
                pre_testx, (3, image_size*image_size*3))
            end_index = start_index + 3
        test_x[start_index: end_index, :]=pre_testx
        test_y[start_index : end_index, :] = pre_testy

        if count == 16:
            break

    traindata = np.double(train_x/255)
    trainlabel = np.double(train_y)
    testdata = np.double(test_x/255)
    testlabel = np.double(test_y)

    print("------BLS_BASE------")
    bls = BaseBls(N1, N2, N3, s, C)
    start_time = time.time()
    train_out = bls.forward(traindata, trainlabel)
    end_time = time.time()
    train_time = end_time - start_time
    train_acc = show_mnist_accuracy(train_out, trainlabel)
    print('Training accurate is', train_acc*100, '%')
    print('Training time is ', train_time, 's')

    bls.eval()
    start_time = time.time()
    predict_out = bls.forward(testdata)
    end_time = time.time()
    test_time = end_time - start_time
    test_acc = show_mnist_accuracy(predict_out, testlabel)
    print('Testing accurate is', test_acc * 100, '%')
    print('Testing time is ', test_time, 's')
    return

if __name__ == "__main__":
    # test_mnist("./datasets/mnist.mat")
    test_messidor()
