import time
import numpy as np
from sklearn import preprocessing
from scipy import linalg as LA
import math

class BaseBls:
    def __init__(self, N1, N2, N3, S, C, train_num, val_num):
        self.N1 = N1
        self.N2 = N2
        self.N3 = N3
        self.S = S
        self.C = C
        self.train_num = train_num
        self.val_num = val_num
        self.train = True
        self.train_index = 0
        self.val_index = 0
        self.temp_feature_weight_list = []
        self.feature_weight_list = []
        self.dist_maxmin = []
        self.each_window_min = []

    def eval(self):
        self.train = False
        return


    def shrinkage(self, a, b):
        z = np.maximum(a - b, 0) - np.maximum( -a - b, 0)
        return z


    def tansig(self, x):
        return (2/(1+np.exp(-2*x)))-1

    def sparse_autoencoder(self, z, x, batch_size, lam=0.001, iters=50):
        AA = z.T.dot(z)   
        m = z.shape[1]
        n = x.shape[1]
        x1 = np.zeros([m, n])
        wk = x1
        ok = x1
        uk = x1
        L1 = np.mat(AA + np.eye(m)).I
        L2 = np.zeros([m, n])
        index = 0
        for i in range(math.ceil(self.train_num / batch_size)):
            if i != math.ceil(self.train_num / batch_size) - 1:
                z_temp = z[index:index+batch_size, :]
                x_temp = x[index:index+batch_size, :]
                L2 += (L1.dot(z_temp.T)).dot(x_temp)
                index += batch_size
            elif i == math.ceil(self.train_num / batch_size) - 1:
                z_temp = z[index:, :]
                x_temp = x[index:, :]
                L2 += (L1.dot(z_temp.T)).dot(x_temp)
                
        for i in range(iters):
            ck = L2 + np.dot(L1, (ok - uk))
            temp_val = ck + uk
            ok = self.shrinkage(temp_val, lam)
            uk = uk + ck - ok
            wk = ok
        return wk.T

    def standardization(self, x, index):
        self.dist_maxmin.append(np.max(x, axis=0) - np.min(x, axis=0))
        self.each_window_min.append(np.min(x, axis=0))
        return (x - self.each_window_min[index]) / self.dist_maxmin[index]

    def get_enhance_weight(self, N1, N2, N3):
        if N1 * N2 >= N3:
            np.random.seed(67797325)
            return LA.orth(2 * np.random.randn(N2 * N1 + 1, N3)) - 1
        else:
            np.random.seed(67797325)
            return LA.orth(2 * np.random.randn(N2 * N1 + 1, N3).T - 1).T
        
    def pinv(self, A):
        return np.mat(self.C*np.eye(A.shape[1])+A.T.dot(A)).I.dot(A.T)
        
    def get_output_weight(self, A, Y):
        A_pinv = self.pinv(A)
        W = np.dot(A_pinv, Y)
        return W

    def trainpre_process(self, train_x, batch_size, flag):
        train_x = preprocessing.scale(train_x, axis=1)
        x_bias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0], 1))])
        if self.train_index == 0:
            self.temp_mlayer_output = np.zeros([self.train_num, self.N2 * self.N1])
            self.mapping_layer_output = np.zeros([self.train_num, self.N2 * self.N1])
            self.pre_trainx = np.zeros([self.train_num, train_x.shape[1]+1])

        for i in range(self.N2):
            if flag:
                np.random.seed(i)
                if self.train_index == 0:
                    feature_weight = 2 * np.random.randn(train_x.shape[1]+1, self.N1) - 1
                    self.temp_feature_weight_list.append(feature_weight)
                else:
                    feature_weight = self.temp_feature_weight_list[i]
                feature_window = np.dot(x_bias, feature_weight)
                self.temp_mlayer_output[self.train_index:self.train_index+batch_size, self.N1*i:self.N1*(i+1)] = feature_window
            else:
                feature_node_output = np.dot(x_bias, self.feature_weight_list[i])
                self.mapping_layer_output[self.train_index:self.train_index+batch_size, self.N1*i:self.N1*(i+1)] = feature_node_output
        
        if flag:
            self.pre_trainx[self.train_index:self.train_index+batch_size, :] = x_bias

        self.train_index += batch_size
        if (self.train_index == self.train_num) and flag:
            scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(self.temp_mlayer_output)
            self.preprocess_feature = scaler1.transform(self.temp_mlayer_output)
            self.train_index = 0
        elif (self.train_index == self.train_num) and (not flag):
            for i in range(self.N2):
                self.mapping_layer_output[:, self.N1*i:self.N1*(i+1)] = self.standardization(self.mapping_layer_output[:, self.N1*i:self.N1*(i+1)], i)
                self.train_index = 0
        return

    def get_se_weight(self, batch_size): 
        for i in range(self.N2):
            sparse_feature_weight = self.sparse_autoencoder(self.preprocess_feature[:, self.N1*i:self.N1*(i+1)], self.pre_trainx, batch_size)
            self.feature_weight_list.append(sparse_feature_weight)

        return

    def train_forward(self, train_y):
        enhance_layer_input = np.hstack([self.mapping_layer_output, 0.1 * np.ones((self.mapping_layer_output.shape[0], 1))])
        self.enhance_weight = self.get_enhance_weight(self.N1, self.N2, self.N3)
        temp_enhance_output = np.dot(enhance_layer_input, self.enhance_weight)
        self.shrink_para = self.S / np.max(temp_enhance_output)
        enhance_layer_output = self.tansig(temp_enhance_output * self.shrink_para)

        self.output_layer_input = np.hstack([self.mapping_layer_output, enhance_layer_output])
        self.output_weight = self.get_output_weight(self.output_layer_input, train_y)
        
        output_train = np.dot(self.output_layer_input, self.output_weight)
        return output_train

    def evalpre_process(self, test_x, batch_size):
        test_x = preprocessing.scale(test_x, axis=1)
        x_bias = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0], 1))])
        if self.val_index == 0:
            self.eval_mapping_layer_output = np.zeros([self.val_num, self.N2 * self.N1])

        for i in range(self.N2):
            feature_node_output = np.dot(x_bias, self.feature_weight_list[i])
            #feature_node_output = self.standardization(feature_node_output, i)
            self.eval_mapping_layer_output[self.val_index:self.val_index+batch_size, self.N1*i:self.N1*(i+1)] = feature_node_output

        self.val_index += batch_size
        if self.val_index == self.val_num:
            for i in range(self.N2):
                self.eval_mapping_layer_output[:, self.N1*i:self.N1*(i+1)] = self.standardization(self.eval_mapping_layer_output[:, self.N1*i:self.N1*(i+1)], i)
                self.train_index = 0            
        return

    def eval_forward(self):
        enhance_layer_input = np.hstack([self.eval_mapping_layer_output, 0.1 * np.ones((self.eval_mapping_layer_output.shape[0], 1))])
        temp_enhance_output = np.dot(enhance_layer_input, self.enhance_weight)
        enhance_layer_output = self.tansig(temp_enhance_output * self.shrink_para)

        output_layer_input = np.hstack([self.eval_mapping_layer_output, enhance_layer_output])
        output_eval = np.dot(output_layer_input, self.output_weight)

        return output_eval

    def forward(self, x=None, y=None):
        if self.train:
            return self.train_forward(y)
        else:
            return self.eval_forward()

    # def add_enhanceNode(self, iters, add_num):
    #     self.add_shrink_para = []
    #     for i in range(iters):
    #         new_enhance_weight = self.get_enhance_weight()

    #     temp_enhance_output = np.dot(self.mapping_layer_output, new_enhance_weight)
    #     self.shrink_para = self.S / np.max(temp_enhance_output)
    #     add_enhance_layer_output = self.tansig(temp_enhance_output * self.shrink_para)        
        
    #     temp_output_layer_input = np.hstack([self.output_layer_input, add_enhance_layer_output])

    #     D = 