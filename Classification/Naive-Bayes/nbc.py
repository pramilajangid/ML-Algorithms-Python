import pandas as pd
import numpy as np 
from collections import defaultdict
import scipy.stats as s

class NBC:
    """I tell you which class to choose, ok."""

    def __init__(self, data, class_col_name, means_dict, variances_dict, prior_same=True, multiple_col=True):
        self.multiple_col = multiple_col
        self.data = data
        self.class_col_name = class_col_name
        self.means = means_dict
        self.variances = variances_dict
        self.classes = list(data[class_col_name].unique())
        self.prior_same = prior_same
        
    
    def calculate(self):
        data = self.data
        data = data.drop([self.class_col_name], axis=1)
        
        self.predicted_class = defaultdict(int)
        
        if self.multiple_col :
            for k in range(data.shape[0]):
                
                dict_of_likelihood_pb = defaultdict(int)
                for i in self.classes:
                    dict_of_likelihood_pb[i] = (s.multivariate_normal.pdf(data.iloc[k,:], mean=self.means[i], cov=self.variances[i]))
                    
                if not (self.prior_same): 
                    
                    self.prior_pb = dict(self.data[self.class_col_name].value_counts()/self.data.shape[0])
                    for i in dict_of_likelihood_pb:
                        dict_of_likelihood_pb[i] = dict_of_likelihood_pb[i] * self.prior_pb[i]
        
                
                max_val = max(dict_of_likelihood_pb.values())
                
                
                for i in dict_of_likelihood_pb:
                    
                    if dict_of_likelihood_pb[i] == max_val:
                        self.predicted_class[k] = i
                
            return self.predicted_class
        
        else :
            
            for k in range(data.shape[0]):

                dict_of_likelihood_pb = defaultdict(int)
                
                for i in self.classes:
                    
                    dict_of_likelihood_pb[i] = (s.norm.pdf(data.iloc[k,:], loc=self.means[i], scale=self.variances[i]))

                if not (self.prior_same): 

                    self.prior_pb = dict(self.data[self.class_col_name].value_counts()/self.data.shape[0])
                    
                    for i in dict_of_likelihood_pb:
                        
                        dict_of_likelihood_pb[i] = dict_of_likelihood_pb[i] * self.prior_pb[i]

                
                max_val = max(dict_of_likelihood_pb.values())


                for i in dict_of_likelihood_pb:

                    if dict_of_likelihood_pb[i] == max_val:
                        
                        self.predicted_class[k] = i
                        
            return self.predicted_class