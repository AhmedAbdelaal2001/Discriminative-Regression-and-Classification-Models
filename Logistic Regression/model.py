import numpy as np
from utility import add_intercept

class LogisticRegression:
    
    def __init__(self, step_size = 0.01, epsilon = 1e-5):
        self.w = None
        self.step_size = step_size
        self.epsilon = epsilon
        
    def predict(self, X, addIntercept = False):
        if self.w.any() == None:
            raise Exception("Error. Model not trained yet.")
        
        if addIntercept:
            X = add_intercept(X)
        
        z = np.matmul(self.w, np.transpose(X))
        return 1 / (1 + np.exp(-1 * z))
    
    def predict_binary(self, X, addIntercept = False):
        
            return (self.predict(X, addIntercept) >= 0.5).astype(np.int32)
    
    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        
        while True:
            y_pred = self.predict(X)
            gradient = ((y_pred - y) * np.transpose(X)).mean(axis = 1)
            hessian = np.matmul((y_pred * (1 - y_pred)) * np.transpose(X), X) / X.shape[0]
            difference = np.matmul(gradient, np.linalg.inv(hessian))
            self.w -= difference
            print(f"theta = {self.w}")
            
            if np.linalg.norm(difference) < self.epsilon:
                print("Training Complete.")
                return self
