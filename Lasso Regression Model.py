import numpy as np

# Lasso Regression

# Creating a class for Lasso Regression

class Lasso_Regression():
    
    # initialing the hyper parameters
    def __init__(self, learning_rate, no_of_iterations, lambda_parameter):
          self.learning_rate = learning_rate
          self.no_of_iterations = no_of_iterations
          self.lambda_parameter = lambda_parameter
          
    # Fitting the dataset to the Lasso Regression model
    def fit(self, X, Y):
        self.m, self.n = X.shape
        
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y
        
        #implementing Gradient Descent algorithm for optimization
        
        for i in range(self.no_of_iterations):
            self.update_weights()

    # Updating the weight and bias values
    def update_weights(self):
        
        # linear equation of the model
        Y_prediction = self.predict(self.X)
        
        # gradients (dw, db)
        dw = np.zeros(self.n)
        
        
        for i in range(self.n):
            
            if self.w[i] > 0:
                dw[i] = (-(2 * (self.X[:, i ]).dot(self.Y - Y_prediction)) + self.lambda_parameter) / self.m 
            else:
                dw[i] = (-(2 * (self.X[:, i ]).dot(self.Y - Y_prediction)) - self.lambda_parameter) / self.m 
        
        # gradient for bias         
        db = -2 * np.sum(self.y - Y_prediction) / self.m
        
        # updating weight and bias
        self.w = self.w - self.learning_rate * dw 
        
        self.b = self.b - self.learning_rate * db 
    
    # predicting the target variable
    def predict(self, X):
        return X.dot(self.w) + self.b