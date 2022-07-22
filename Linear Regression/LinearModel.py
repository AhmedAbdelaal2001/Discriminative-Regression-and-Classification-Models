import numpy as np

class LinearRegression:
    
    def __init__(self, w = None):
        
        self.w = w
    
    
    
    def fit(self, X, y):
        
        self.w = np.linalg.solve(np.transpose(X) @ X, np.transpose(X) @ y)



    def predict(self, X):
        
        if self.w.any() == None:
            raise Exception("Error! Model not trained yet.")
        
        return np.dot(self.w, X)