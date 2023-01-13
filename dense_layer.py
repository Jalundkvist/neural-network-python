import random

class Dense_layer:
    
    """ Dense_layer class, to be used in a neural network for easy set-up of new layers.
    """    
    def __init__(self, num_nodes, num_weights) -> None:
        self.error = []
        self.bias = []
        self.weights = [] 
        self.output = []
        self.num_nodes = 0
        self.num_weights = 0
        self.set_nodes(num_nodes, num_weights)

    def set_nodes(self, num_nodes, num_weights):
        """set_nodes sets the start value on
        weights and biases for each node in the given layer

        Parameters
        ----------
        num_nodes
            number of nodes in current layer
        num_weights
            number of weights (number of nodes - previous layer)
        """        
        self.num_nodes = num_nodes
        self.num_weights = num_weights
        self.error = [0] * num_nodes
        self.output = [0] * num_nodes
        for _ in range(num_nodes):
            self.bias.append(round(random.uniform(0, 1),2))
            self.weights.append([round(random.uniform(0, 1),2) for _ in range(num_weights)])

    def print(self, precision):
        """print the dense_layers values, such as output, error, bias, weights and the
        number of nodes and weights per node.

        Parameters
        ----------
        precision
            number of decimals
        """        
        self.round(precision)
        print("--------------------------------------------------------")
        print(f"Number of nodes:\t\t{self.num_nodes}")
        print(f"Number of weights per node:\t{self.num_weights}")
        print(f"Output:\t\t\t\t{self.rounded_output}")
        print(f"Error:\t\t\t\t{self.rounded_error}")
        print(f"Bias:\t\t\t\t{self.rounded_bias}")
        print(f"Weights:")
        for i in range(self.num_nodes):
            print(f"\t\t\tNode {i+1}:\t{self.rounded_weights[i]}") 
        print("--------------------------------------------------------")
        
    def round(self, precision):
        """round all the values to number of decimals

        Parameters
        ----------
        precision
            number of decimals
        """   
        self.rounded_bias =  [round(x,precision) for x in self.bias]
        self.rounded_weights = [[round(x,precision) for x in sublist] for sublist in self.weights]
        self.rounded_output =  [round(x,precision) for x in self.output]
        self.rounded_error = [round(x,precision) for x in self.error]

    def delta_relu(self, x):
        if x > 0:
            return 1
        else:
            return 0

    def relu(self, output):
        if output > 0:
            return output
        else:
            return 0

    def feedforward(self, input):
        for i in range(self.num_nodes):
            sum = self.bias[i]
            for j in range(self.num_weights):
                sum += input[j] * self.weights[i][j]
            self.output[i] = self.relu(sum)

    def backprop_outer(self, reference):
        for i in range(len(self.output)):
            dev = reference[i] - self.output[i]
            self.error[i] = dev #* self.delta_relu(self.output[i]) # Commented since it removes any error from the output layer.
                                # IMHO this should not have RELU.
         
    def backprop_hidden(self, next_layer):
        for i in range(len(self.output)):
            sum = 0
            for index, error in enumerate(next_layer.error):
                sum += error * next_layer.weights[index][i]
                self.error[i] = sum * self.delta_relu(self.output[i]) 
    
    def optimize(self, input, learn_rate):
            for i in range(len(self.output)):
                self.bias[i] += self.error[i] * learn_rate
                for j in range(self.num_weights): 
                    self.weights[i][j] += self.error[i] * learn_rate * input[j]

if __name__ == "__main__":
    print("Not the main file")