import numpy as np

class mlr:
    ''' Pass in the target column encoded using one-hot-encoding.
    '''

    def __init__(self, number_of_classes, X):
        self.number_of_classes = number_of_classes
        self.X = X
    
    def __softmax(self, theta0, theta):
        ''' an array containing softmax function value for each class'''
        
        k = self.number_of_classes
        X = self.X
        numerator = np.array()
        denominator = 0
        for i in range(k):
            power_term = np.exp(theta0[:, i] + np.matmu(X, (theta[:, i])))

            np.append(numerator, power_term , axis=0)

            denominator += power_term

        return numerator/denominator

    def __neg_log_likelihood (self, theta0, theta):
        np.log(__softmax(theta0, theta))






