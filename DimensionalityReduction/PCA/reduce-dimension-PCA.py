#script for dimensionality reduction

import numpy as np

class der:
    
    def reduce_dimensions(data, variance_to_preserve):
        
        """data should be a dataframe and provided without class column and should already be pre-processed(SMOTE applied, if SMOTE is neccessary and definitely be converted to zero mean)"""
        
        cov_matrix = np.array(data.cov())
        
        factorised_matrix = np.linalg.svd(cov_matrix)
        
        total_eigen_values = np.sum(factorised_matrix[1])
        
        sum_of_eigen_values = 0
        
        l = 1
        
        for i in factorised_matrix[1] :
            
            if (sum_of_eigen_values/total_eigen_values) > variance_to_preserve :
                
                break
             
            sum_of_eigen_values = sum_of_eigen_values + i
            
            l = l + 1
       
        matrix_q = factorised_matrix[0][:,:l]
        
        reduced_data = np.matmul(np.array(data), np.array(matrix_q))
        
        return reduced_data