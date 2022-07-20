import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def add_intercept(X):
    new_X = np.zeros((X.shape[0], X.shape[1] + 1), dtype = X.dtype)
    new_X[:, 0] = 1
    new_X[:, 1:] = X
    
    return new_X

def load_dataset(filename, addIntercept = False):
    dataset = pd.read_csv(filename)
    
    X = dataset.iloc[:, :-1].values
    if addIntercept:
        X = add_intercept(X)
    
    y = dataset.iloc[:, -1].values
    
    return X, y

def plot(X, y, x_min, y_min, x_max, y_max, model = None):
    plt.figure()
    
    if model is not None:
        plt.plot(X[y == 1, 0], X[y == 1, 1], 'bx')
        plt.plot(X[y == 0, 0], X[y == 0, 1], 'go')
        x1 = np.arange(min(X[:, 0]), max(X[:, 1]), 0.01)
        x2 = -1 * (model.w[0] / model.w[2] + model.w[1] / model.w[2] * x1)
        plt.plot(x1, x2, c = "red")
    
    else:
        plt.plot(X[y == 1, 1], X[y == 1, 2], 'bx')
        plt.plot(X[y == 0, 1], X[y == 0, 2], 'go')
        
    plt.xlabel("x1")
    plt.ylabel("x2")
    
    plt.xlim(xmin = x_min, xmax = x_max)
    plt.ylim(ymin = y_min, ymax = y_max)
      
    plt.show()
    
def accuracy(y_pred, y_true):
    count = 0
    
    for i in range(len(y_true)):
        if y_pred[i] == y_true[i]:
            count += 1
    
    return count / len(y_true)

def save_parameters(model1, model2):
    np.savetxt("parameters1.txt", model1.w, delimiter = ", ")
    np.savetxt("parameters2.txt", model2.w, delimiter = ", ")
    
    print("Parameters Saved.")
    print(f"Model 1 parameters: {model1.w}")
    print(f"Model 2 parameters: {model2.w}")