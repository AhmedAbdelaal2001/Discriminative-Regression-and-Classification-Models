import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def add_intercept(X):
    X_new = np.zeros([X.shape[0], X.shape[1] + 1], dtype = X.dtype)
    X_new[:, 0] = 1
    X_new[:, 1:] = X
    
    return X_new

def load_dataset(filename, addIntercept = False):
    
    dataset = pd.read_csv(filename)
    X = dataset.iloc[:, :-1].values
    
    if addIntercept:
        X = add_intercept(X)
        
    y = dataset.iloc[:, -1].values
    
    return X,y

def plot(X, y, model):
    x_axis = y
    y_axis = model.predict(add_intercept(X))
    
    plt.figure()
    plt.scatter(x_axis, y_axis)
    
    plt.xlabel("True Count.")
    plt.ylabel("Predicted Count.")
    
    plt.show()
    
def save_parameters(model):
    
    model_parameters = model.w
    np.savetxt("parameters.txt", model_parameters, delimiter = ", ")
    print(model_parameters)
    print("Parameters Saved.")