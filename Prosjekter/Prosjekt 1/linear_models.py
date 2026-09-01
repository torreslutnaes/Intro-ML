import numpy as np
import pandas as pd

class LinearRegression():
    def __init__(self, lr=0.001, n_iterations=1000):
        self.weights = None
        self.bias = None
        
        self.lr = lr
        self.n_iterations = n_iterations

        self.train_accuracies = []
        self.loss_history = []
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        # ==================================== 
        self.weights = np.zeros(X.shape[1]) 
        self.bias = 0.0

        for i in range(self.n_iterations):
            lin_model = np.matmul(self.weights, X.transpose()) + self.bias #y_pred

            error = lin_model - y
            m = X.shape[0]

            #compute_parameters() fra kompendiet

            dw = (2/m) * np.matmul(X.transpose(), error)
            db = (2/m) * np.sum(error)

            #update_parameters() fra kompendiet

            self.weights -= self.lr * dw
            self.bias -= self.lr * db


            
        # ====================================
        # raise NotImplementedError("LinearRegression.fit is not implemented yet.")
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        # ====================================
        return np.matmul(self.weights, X.transpose()) + self.bias 
        # ====================================
        # raise NotImplementedError("LinearRegression.predict is not implemented yet.")
    
class LogisticRegression():
    def __init__(self, lr=0.001, n_iterations=1000):
        self.weights = None
        self.bias = None
        
        self.lr = lr
        self.n_iterations = n_iterations
        
        self.loss_history = []
        self.train_accuracies = []

    def sigmoid(self, z):
        # ====================================
        return 1 / (1 + np.exp(-z))
        # ====================================
        # raise NotImplementedError("LogisticRegression.sigmoid is not implemented yet.")

    def compute_loss(self, y, y_pred):
        return (-y*np.log(y_pred) - ((1-y)*np.log(1-(y_pred)))).mean()

    def compute_gradients(self, x, y, y_pred):
        grad_w = ((y_pred-y)*x.transpose()).mean(axis=1)
        grad_b = (y_pred - y).mean()
        return grad_w, grad_b
    
    def update_parameters(self, grad_w, grad_b):
        self.weights -= self.lr*grad_w
        self.bias -= self.lr*grad_b

    def accuracy(self, true_values, predictions):
        return np.mean(true_values == predictions)
    
    
    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1]) 
        self.bias = 0.0
        for i in range(self.n_iterations):
            lin_model = np.matmul(self.weights, X.transpose()) + self.bias
            y_pred = self.sigmoid(lin_model)
            grad_w, grad_b = self.compute_gradients(X, y, y_pred)
            self.update_parameters(grad_w, grad_b)

            loss = self.compute_loss(y, y_pred)
            pred_to_class = [1 if _y > 0.5 else 0 for _y in y_pred]
            self.train_accuracies.append(self.accuracy(y, pred_to_class))
            self.loss_history.append(loss)
        # raise NotImplementedError("LogisticRegression.fit is not implemented yet.")
    
    def predict_proba(self, X):
        # ====================================
        lin_model = np.matmul(X, self.weights) + self.bias
        y_pred = self.sigmoid(lin_model)
        return [1 if _y > 0.5 else 0 for _y in y_pred]
        # ====================================
        # raise NotImplementedError("LogisticRegression.predict_proba is not implemented yet.")

    

