import scipy.io as sio
from sklearn.model_selection import train_test_split
import numpy as np
import torch

def get_train_test_data_from_mat(dataname, test_size):
    current_data = sio.loadmat(dataname)
    X = current_data['data']
    Y = current_data['label']                              # 1000 for usps pendigits, 900 for abalone
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=100, shuffle=True)
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    Y_train = torch.from_numpy(Y_train).float()
    Y_test = torch.from_numpy(Y_test).float()
    #y_test = Y_test.argmax(dim=1)
    return X_train, Y_train, X_test, Y_test # note that Y_train is matrix, while y_test is vector
