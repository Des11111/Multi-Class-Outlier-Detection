import numpy as np

def generate_data_uniform_plus_normal(n_in_1, n_in_2, n_out, dim=50, mean_c1=1, mean_c2=-1, radius=4, a_signal=8):
    
    n = n_in_1 + n_in_2 + n_out
    assert (n>0)
    
    X = np.zeros((0, dim))
    Y = np.zeros((0,))
    
    if n_in_1 > 0:
        inliers_type1 = np.random.uniform(low=mean_c1 - radius, high=mean_c1 + radius, size=(n_in_1, dim)) \
                    + np.random.normal(loc=0, scale=1, size=(n_in_1, dim))
        X = np.concatenate([X,inliers_type1], 0)
        Y = np.concatenate([Y,1*np.ones((n_in_1,))], 0)
        
    if n_in_2 > 0:
        inliers_type2 = np.random.uniform(low=mean_c2 - radius, high=mean_c2 + radius, size=(n_in_2, dim)) \
                    + np.random.normal(loc=0, scale=1, size=(n_in_2, dim))
        X = np.concatenate([X,inliers_type2], 0)
        Y = np.concatenate([Y,2*np.ones((n_in_2,))], 0)
        
    if n_out > 0:
        outlier_c1 = np.random.uniform(low=mean_c1 - radius, high=mean_c1 + radius, size=(n_out, dim)) \
                               + a_signal * np.random.normal(loc=0, scale=1, size=(n_out, dim))
        outlier_c2 = np.random.uniform(low=mean_c2 - radius, high=mean_c2 + radius, size=(n_out, dim)) \
                               + a_signal * np.random.normal(loc=0, scale=1, size=(n_out, dim))

        outliers = np.vstack((outlier_c1,outlier_c2))
        np.random.shuffle(outliers)
        outliers = outliers[0:n_out]
        X = np.concatenate([X,outliers], 0)
        Y = np.concatenate([Y,np.zeros((n_out,))], 0)
    
    return X,Y.astype(int)
    