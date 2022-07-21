import numpy as np
from utility import add_intercept

class PoissonRegression:
    
    def __init__(self, w = None, alpha = 1e-5, epsilon = 1e-5):
        self.w = w
        self.alpha = alpha
        self.epsilon = epsilon
        
    def predict(self, X, addIntercept = False):
        if self.w.any() == None:
            raise Exception("Error! Model Not Trained Yet.")
        
        if addIntercept:
            X = add_intercept(X)
            
        z = np.matmul(self.w, np.transpose(X))
        return np.exp(z)
    
    def fit(self, X, y):
        self.w = np.zeros([X.shape[1]])
        
        while True:
            difference = ((y - self.predict(X)) * np.transpose(X)).mean(axis = 1)
            self.w += self.alpha * difference
            print(self.w)
            
            if np.linalg.norm(difference) < self.epsilon:
                print("Training Complete.")
                break