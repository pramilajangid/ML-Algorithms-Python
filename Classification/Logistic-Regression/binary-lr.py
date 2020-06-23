import numpy as np

class binary_lr :
    '''Finds updated parameters, using gradient descent algorithm.
       If there are n examples and m features then our X will be nxm. 
       C will be nx1.
    '''

    def __init__(self, step_size, X, C):
        self.step_size = step_size
        self.X = X
        self.C = C

    def __the_sigmoid_output(self, theta0, theta):
        '''Calculates the whole posterior probability array for each example. 
           X will be nxm.
           theta0 will be 1x1.
           theta will be 1xm.
        '''
        X = self.X

        exp_power = theta0 + (np.matmul(theta, X.T)) # array of 1xn
        
        denominator = 1 + np.exp(-exp_power) # array of 1xn
        
        return 1/denominator # array of 1xn

    def __neg_log_likelihood(self, theta0, theta):
        '''X will be nxm.
           theta0 will be 1x1.
           theta will be 1xm.

           returns an array of 1x1 containing value of negative log likelihood 
           value, for given thetas
        '''
        sigmoid_output = __the_sigmoid_output(theta0, theta) # array of 1xn
        
        first_term = np.matmul(np.log(sigmoid_output), C) # array of 1x1, where C is nx1
        
        second_term = np.matmul(np.log(1 - sigmoid_output), (1-C)) # array of 1x1
            
        return first_term + second_term # array of 1x1

    def __derivative_theta0(self, theta0, theta):
        '''Returns derivative with respect to theta0 as an array of 1x1
           X will be nxm.
           theta0 will be 1x1.
           theta will be 1xm.
           C will be nx1
        '''
        sigmoid_output = __the_sigmoid_output(theta0, theta) # array of 1xn
        
        return np.sum(C - sigmoid_output.T) # array of nx1 where C is 1xn

    def __derivative_theta(self, theta0, theta):
        '''Returns derivative with respect to theta 
           X will be nxm.
           theta0 will be 1x1.
           theta will be 1xm.
        '''
        X = self.X  # nxm

        sigmoid_output = __the_sigmoid_output(theta0, theta) # 1xn 
        
        return np.matmul((C.T - sigmoid_output) * X) # C.T - sigmoid_output is 1xn and X 
                                                     # is nxm  resulting in gradient vector
                                                     # of 1xm

    def update_parameters(self, theta0_initial, theta_initial): 
        
        derivative_theta0 = __derivative_theta0(theta0_initial, theta_initial) # array of 1x1
        derivative_theta = __derivative_theta(theta0_initial, theta_initial) # array of 1xm
        
        theta0_final = theta0_initial + (step_size * derivative_theta0)
        theta_final = theta_initial + (step_size* derivative_theta) 

        return theta0_final, theta_final # theta_final => 1xm ; theta0_final => 1x1
    