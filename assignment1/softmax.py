#-*- coding:utf-8 -*-

import numpy as np


def softmax_loss_naive(W, X, y, reg):    

    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)    # 得到一个和W同样shape的矩阵
    dW_each = np.zeros_like(W)
    num_train, dim = X.shape
    num_class = W.shape[1]
    f = X.dot(W)    # N by C
    # Considering the Numeric Stability
    f_max = np.reshape(np.max(f, axis=1), (num_train, 1))   # 找到最大值然后减去，这样是为了防止后面的操作会出现数值上的一些偏差
    prob = np.exp(f - f_max) / np.sum(np.exp(f - f_max), axis=1, keepdims=True) # N by C
    y_trueClass = np.zeros_like(prob)
    y_trueClass[np.arange(num_train), y] = 1.0
    for i in range(num_train):
        for j in range(num_class):
            loss += -(y_trueClass[i, j] * np.log(prob[i, j]))    # 损失函数的公式L = -(1/N)∑i∑j1(k=yi)log(exp(fk)/∑j exp(fj)) + λR(W)
            dW_each[:, j] = -(y_trueClass[i, j] - prob[i, j]) * X[i, :]    # 梯度的公式 ∇Wk L = -(1/N)∑i xiT(pi,m-Pm) + 2λWk, where Pk = exp(fk)/∑j exp(fj
        dW += dW_each     # 这是把每个类的放在了一起
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)  # 加上正则
    dW /= num_train
    dW += reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):    
    """    
    Softmax loss function, vectorized version.    

    Inputs and outputs are the same as softmax_loss_naive.    
    """    
    # Initialize the loss and gradient to zero.    
    loss = 0.0    
    dW = np.zeros_like(W)    # D by C    
    num_train, dim = X.shape

    f = X.dot(W)    # N by C
    # Considering the Numeric Stability
    f_max = np.reshape(np.max(f, axis=1), (num_train, 1))   # N by 1
    prob = np.exp(f - f_max) / np.sum(np.exp(f - f_max), axis=1, keepdims=True)
    y_trueClass = np.zeros_like(prob)
    y_trueClass[range(num_train), y] = 1.0    # N by C
    loss += -np.sum(y_trueClass * np.log(prob)) / num_train + 0.5 * reg * np.sum(W * W)#向量化直接操作即可
    dW += -np.dot(X.T, y_trueClass - prob) / num_train + reg * W

    return loss, dW