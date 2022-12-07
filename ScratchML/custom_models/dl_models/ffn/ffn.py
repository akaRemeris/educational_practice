import numpy as np
from utils.activation import * 
from utils.loss import *

class Custom_FFN(object):
    def __init__(self, architecture, 
            activation_function=sigmoid, activation_function_derivative=sigmoid_derivative,
            last_activation_function=sm, loss_function=cross_entropy_loss, loss_function_derivative=dCE_dLinear):

        self.parameters = {'weights': [], 'biases': []}
        self.activation_functions_values = []
        self.activation_function_derivative_values = []
        for layer_id in range(len(architecture)-1):

            weights = np.random.uniform(size=(architecture[layer_id], architecture[layer_id+1]))
            self.parameters['weights'].append(weights)

            bias = np.random.uniform(size=(1, architecture[layer_id+1]))
            self.parameters['biases'].append(bias)

        self.n_hidden_layers = len(architecture)-2

        self.activation_function = activation_function
        self.last_activation_function = last_activation_function

        self.activation_function_derivative = activation_function_derivative
                
        self.loss_function = loss_function
        self.loss_function_derivative = loss_function_derivative

   
    def forward(self, X):
        self.activation_functions_values = []
        self.activation_function_derivative_values = []
        
        for param_idx in range(len(self.parameters['weights'])-1):
            X = linear_transformation(X, self.parameters['weights'][param_idx], self.parameters['biases'][param_idx])
            dAct_dLinear = self.activation_function_derivative(X)
            X = self.activation_function(X)            

            self.activation_functions_values.append(X.copy())
            self.activation_function_derivative_values.append(dAct_dLinear)

        X = linear_transformation(X, self.parameters['weights'][-1], self.parameters['biases'][-1])
        X = self.last_activation_function(X)

        return X

