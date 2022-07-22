import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def add_intercept(X):
    X_new = np.zeros([X.shape[0], X.shape[1] + 1])
    
    X_new[:, 0] = 1
    X_new[:, 1:] = X
    
    return X_new

def load_dataset(filename):
    dataset = pd.read_csv(filename)
    
    X = dataset.iloc[:, :-1].values
    X = add_intercept(X)
    
    y = dataset.iloc[:, -1].values
    
    return X, y

def add_polynomialTerms(X, k):
    
    X_new = np.ones([X.shape[0], k + 1])
    X_new[:, :X.shape[1]] = X
    
    for i in range(X.shape[1], k + 1):
        X_new[:, i] = X[:, 1] ** i
        
    return X_new

def add_sine_poly(X, k):
    
    X_new = np.ones([X.shape[0], k + 2])
        
    if k != 0:
        
        if k != 1:
            poly_shape = X.shape[1] - 1
        else:
            poly_shape = X.shape[1]
            
        
        X_new[:, :poly_shape] = X[:, :poly_shape]
    
        for i in range(poly_shape, k + 1):
            X_new[:, i] = X[:, 1] ** i
        
        X_new[:, k + 1] = X[:, X.shape[1] - 1]
        
        
    if k != 0 and k != 1:
        X_new[:, k + 1] = X[:, X.shape[1] - 1]
    else:
        X_new[:, k + 1] = np.sin(X[:, 1])
    
    return X_new

def plot(X, y, title, model = None, k = 1, sine = False):
    
    plt.figure()
    plt.scatter(X[:, 1], y)
    
    if model is not None:
        x_axis = np.arange(-2 * np.pi, 2 * np.pi, 0.01)
        y_axis = np.zeros(shape = x_axis.shape)
    
        for i in range(k + 1):
            y_axis += model.w[i] * x_axis ** i
        
        if sine:
            y_axis += model.w[k + 1] * np.sin(x_axis)
        
        plt.plot(x_axis, y_axis)
    
    plt.title(title)
    plt.show()