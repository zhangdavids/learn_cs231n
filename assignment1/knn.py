#-*- coding:utf-8 -*-

import numpy as np


class KNearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loops(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %s for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_no_loops(self, X):
        pass
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy.                #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        dists = np.multiply(np.dot(X,self.X_train.T),-2) 
        sq1 = np.sum(np.square(X),axis=1,keepdims = True) 
        sq2 = np.sum(np.square(self.X_train),axis=1) 
        dists = np.add(dists,sq1) 
        dists = np.add(dists,sq2) 
        dists = np.sqrt(dists)
        #########################################################################
        #                         END OF YOUR CODE                              #
        #########################################################################
        return dists

    def compute_distances_one_loops(self, X):
        pass
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
          #######################################################################
          # TODO:                                                               #
          # Compute the l2 distance between the ith test point and all training #
          # points, and store the result in dists[i, :].                        #
          #######################################################################
          dists[i,:] = np.sqrt(np.sum(np.square(self.X_train-X[i,:]),axis = 1)) 
          #######################################################################
          #                         END OF YOUR CODE                            #
          #######################################################################
        return dists

    def compute_distances_two_loops(self, X):
        pass
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
          for j in range(num_train):
            dists[i][j] = np.sqrt(np.sum(np.square(self.X_train[j,:] - X[i,:])))
            #####################################################################
            # TODO:                                                             #
            # Compute the l2 distance between the ith test point and the jth    #
            # training point, and store the result in dists[i, j]. You should   #
            # not use a loop over dimension.                                    #
            #####################################################################
            #####################################################################
            #                       END OF YOUR CODE                            #
            #####################################################################
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.
     
        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.
     
        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i]. 
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
          # A list of length k storing the labels of the k nearest neighbors to
          # the ith test point.
          closest_y = [] 
          ######################################################################### 
          # TODO:                                                                 # 
          # Use the distance matrix to find the k nearest neighbors of the ith    # 
          # training point, and use self.y_train to find the labels of these      # 
          # neighbors. Store these labels in closest_y.                           # 
          # Hint: Look up the function numpy.argsort.                             # 
          ######################################################################### 
          closest_y = self.y_train[np.argsort(dists[i,:])[:k]] 
        
          ######################################################################### 
          # TODO:                                                                 # 
          # Now that you have found the labels of the k nearest neighbors, you    # 
          # need to find the most common label in the list closest_y of labels.   # 
          # Store this label in y_pred[i]. Break ties by choosing the smaller     # 
          # label.                                                                # 
          ######################################################################### 
          y_pred[i] = np.argmax(np.bincount(closest_y))        #########################################################################
          #                           END OF YOUR CODE                            #
          #########################################################################
     
        return y_pred



