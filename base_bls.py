import time
import numpy as np
from sklearn import preprocessing
from scipy import linalg as LA

class BaseBls:
    def __init__(self, N1, N2, N3, S, C):
        self.N1 = N1
        self.N2 = N2
        self.N3 = N3
        self.S = S
        self.C = C
        self.train = True
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

    def sparse_autoencoder(self, z, x, lam=0.001, iters=50):
        AA = z.T.dot(z)   
        m = z.shape[1]
        n = x.shape[1]
        x1 = np.zeros([m, n])
        wk = x1
        ok = x1
        uk = x1
        L1 = np.mat(AA + np.eye(m)).I
        L2 = (L1.dot(z.T)).dot(x)
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

    def train_forward(self, train_x, train_y):
        time1_s = time.time()
        train_x = preprocessing.scale(train_x, axis=1)
        #train_x = np.load("messidor.npy")
        time1_e = time.time()
        print("time1 durations is ", time1_e - time1_s)
        #np.save("messidor.npy", train_x)
        x_bias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0], 1))])
        self.mapping_layer_output = np.zeros([train_x.shape[0], self.N2 * self.N1])

        for i in range(self.N2):
            print("windows : ", i)
            time2_s = time.time()
            np.random.seed(i)
            feature_weight = 2 * np.random.randn(train_x.shape[1]+1, self.N1) - 1
            feature_window = np.dot(x_bias, feature_weight)
            time2_e = time.time()
            print("time2 durations is ", time2_e - time2_s)
            scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(feature_window)
            preprocess_feature = scaler1.transform(feature_window)
            sparse_feature_weight = self.sparse_autoencoder(preprocess_feature, x_bias)
            self.feature_weight_list.append(sparse_feature_weight)
            time3_s = time.time()
            feature_node_output = np.dot(x_bias, sparse_feature_weight)
            time3_e = time.time()
            print("time3 durations is ", time3_e - time3_s)
            feature_node_output = self.standardization(feature_node_output, i)
            self.mapping_layer_output[:, self.N1*i:self.N1*(i+1)] = feature_node_output

        time4_s = time.time()
        enhance_layer_input = np.hstack([self.mapping_layer_output, 0.1 * np.ones((self.mapping_layer_output.shape[0], 1))])
        self.enhance_weight = self.get_enhance_weight(self.N1, self.N2, self.N3)
        temp_enhance_output = np.dot(enhance_layer_input, self.enhance_weight)
        self.shrink_para = self.S / np.max(temp_enhance_output)
        enhance_layer_output = self.tansig(temp_enhance_output * self.shrink_para)
        time4_e = time.time()
        print("time4 durations is ", time4_e - time4_s)
        self.output_layer_input = np.hstack([self.mapping_layer_output, enhance_layer_output])
        time5_s = time.time()
        self.output_weight = self.get_output_weight(self.output_layer_input, train_y)
        
        output_train = np.dot(self.output_layer_input, self.output_weight)
        time5_e = time.time()
        print("time5 durations is ", time5_e - time5_s)
        return output_train

    def eval_forward(self, test_x):
        test_x = preprocessing.scale(test_x, axis=1)
        x_bias = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0], 1))])
        mapping_layer_output = np.zeros([test_x.shape[0], self.N2 * self.N1])

        for i in range(self.N2):
            feature_node_output = np.dot(x_bias, self.feature_weight_list[i])
            feature_node_output = self.standardization(feature_node_output, i)
            mapping_layer_output[:, self.N1*i:self.N1*(i+1)] = feature_node_output

        enhance_layer_input = np.hstack([mapping_layer_output, 0.1 * np.ones((mapping_layer_output.shape[0], 1))])
        temp_enhance_output = np.dot(enhance_layer_input, self.enhance_weight)
        enhance_layer_output = self.tansig(temp_enhance_output * self.shrink_para)

        output_layer_input = np.hstack([mapping_layer_output, enhance_layer_output])
        output_eval = np.dot(output_layer_input, self.output_weight)

        return output_eval

    def forward(self, x, y=None):
        if self.train:
            return self.train_forward(x, y)
        else:
            return self.eval_forward(x)

    # def add_enhanceNode(self, iters, add_num):
    #     self.add_shrink_para = []
    #     for i in range(iters):
    #         new_enhance_weight = self.get_enhance_weight()

    #     temp_enhance_output = np.dot(self.mapping_layer_output, new_enhance_weight)
    #     self.shrink_para = self.S / np.max(temp_enhance_output)
    #     add_enhance_layer_output = self.tansig(temp_enhance_output * self.shrink_para)        
        
    #     temp_output_layer_input = np.hstack([self.output_layer_input, add_enhance_layer_output])

    #     D = 