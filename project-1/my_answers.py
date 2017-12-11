import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))
        
        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        # Sigmoid activation function:
        self.activation_function = lambda x : 1/(1 + np.exp(-x)) 


    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
                features: 2D array, each row is one data record, each column is a feature
                targets: 1D array of target values
        
            Shapes
                features: (batch_size, 56)
                targets: (batch_size, 1)
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        
        for X, y in zip(features, targets):
            
            # Forward pass:
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Backproagation:
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        # Update weights after epoch:    
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)
        
        # Learn rate decat
        self.lr *= 0.9998

    def forward_pass_train(self, X):
        ''' Implement forward pass
         
            Arguments
                X: features batch 
                
            Shapes (Bike-Sharing Dataset)
                X.shape: (56,) 
        '''
        # Layer 1 - Sigmoid activation funtion:
        hidden_inputs = np.matmul(X, self.weights_input_to_hidden)              # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)                # signals from hidden layer
        
        # Layer 2 - No activation funtion:
        final_inputs = np.matmul(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs                                            # signals from final output layer
        
        return final_outputs, hidden_outputs


    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
                final_outputs: output from forward pass
                y: target (i.e. label) batch
                delta_weights_i_h: change in weights from input to hidden layers
                delta_weights_h_o: change in weights from hidden to output layers
            
            Shapes (Bike-Sharing Dataset)
                final_outputs: (1,)
                y: (1,)
                delta_weights_i_h: (56, hidden_nodes)
                delta_weights_h_o: (hidden_nodes, 1)

        '''
        # Output error
        error = y - final_outputs # Output layer error is the difference between desired target and actual output.

        # Hidden layer contribution to the total error:
        hidden_error = np.matmul(error, self.weights_hidden_to_output.T)
        
        # Backpropagated error terms 
        output_error_term = error                                                           # No activation function
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)            # Hidden Error * diff(Sigmoid(Hidden layer output))
        
        # Weight updates due to one row of data
        delta_weights_i_h += np.matmul(X[:,None], hidden_error_term[None,:])                # (input to hidden)
        delta_weights_h_o += np.matmul(hidden_outputs[:,None], output_error_term[None,:])   # (hidden to output)
            
        return delta_weights_i_h, delta_weights_h_o


    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step (per batch)
         
            Arguments
                delta_weights_i_h: change in weights from input to hidden layers
                delta_weights_h_o: change in weights from hidden to output layers
                n_records: number of records
            
            Shapes
                delta_weights_i_h: (56, hidden_nodes)
                delta_weights_h_o: (hidden_nodes, 1)
                n_records: (1,)

        '''
        # Update weights, normalized by the amount of data points used for the batch 
        self.weights_hidden_to_output += self.lr*delta_weights_h_o/n_records                # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr*delta_weights_i_h/n_records                 # update input-to-hidden weights with gradient descent step


    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
                features: 1D array of feature values
            
            Shapes
                feature: (56,)
        '''
        # Layer 1 - Sigmoid activation funtion
        hidden_inputs = np.matmul(features, self.weights_input_to_hidden)               # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)                        # signals from hidden layer
        
        # Layer 2 - No activation function
        final_inputs = np.matmul(hidden_outputs, self.weights_hidden_to_output)         # signals into final output layer
        final_outputs = final_inputs                                                    # signals from final output layer                           
        
        return final_outputs


##########################################################
# Set your hyperparameters here
##########################################################
iterations = 10000
learning_rate = 1.5
hidden_nodes = 5
output_nodes = 1