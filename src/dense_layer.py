import random
import math
class Dense_layer:
    """
    Dense_layer class, to be used in a neural network for easy set-up of new layers.
    It has methods for feedforward, backpropagation, weight optimization, resizing and 
    printing the current values of the layer.
    
    Parameters
    ----------
    num_nodes:
        Number of nodes in the current layer.
    num_weights:
        Number of weights (number of nodes in the previous layer).
        
    Variables
    ----------
    error:
        list containing error values for each node.
    bias:
        list containing bias values for each node.
    weights:
        2d list(array) of all weights for each node.
    output:
        list containing output values for each node.
    """
    def __init__(self, num_nodes, num_weights) -> None:
        self.error = []
        self.bias = []
        self.weights = []
        self.output = []
        self.num_nodes = 0
        self.num_weights = 0
        self.resize(num_nodes, num_weights)
    
    def __del__(self):
        """__del__ 
            The destructor for the Dense_layer class.
        """        
        del self.weights
        del self.bias
        del self.output
        del self.error

    def resize(self, num_nodes, num_weights):
        """resize:
            Updates the values for weights and biases on 
            each node in the given layer.

        Parameters
        ----------
        num_nodes
            number of nodes in current layer.
        num_weights
            number of weights (number of nodes in previous layer).
        """
        self.num_nodes = num_nodes
        self.num_weights = num_weights

        self.error = [0] * num_nodes
        self.output = [0] * num_nodes
        # Generate a bias for each node in the layer
        self.bias = [round(random.uniform(0, 1), 2) for _ in range(num_nodes)]
        # Generate all weights for each node in the layer
        self.weights = [[round(random.uniform(0, 1), 2) for _ in range(num_weights)] for _ in range(num_nodes)]

    def feedforward(self, input):
        """feedforward:
            Takes the input values from either previous layer or and feeds it through
            the nodes in the given layer algorithm and validates the output with
            tanh activation function.

        Parameters
        ----------
        input
            output from previous layer.
        """
        for i in range(self.num_nodes):
            sum = self.bias[i]
            for j in range(self.num_weights):
                sum += input[j] * self.weights[i][j]
            self.output[i] = math.tanh(sum)
        return

    def backprop_outer(self, reference):
        """backprop_outer:
            Calculates error in the outer layer value
            using the tanh activation function.

        Parameters
        ----------
        reference
            reference training data.
        """        
        for i in range(self.num_nodes):
            dev = reference[i] - self.output[i]
            self.error[i] = dev * (1 - math.tanh(self.output[i])**2)
        return

    def backprop_hidden(self, next_layer):
        """backprop_hidden:
            calculates error in the hidden layers
            using the tanh activation function.

        Parameters
        ----------
        next_layer:
            The next hidden layer(counting backwards).
        """    
        for i in range(len(self.output)):
            sum = 0
            for index, error in enumerate(next_layer.error):
                sum += error * next_layer.weights[index][i]
            self.error[i] = sum * (1 - math.tanh(self.output[i])**2)
        return
    
    def optimize(self, input, learn_rate):
        """optimize:
            The bias and weight in the specified layer.

        Parameters
        ----------
        input
            Is used to adjust the weights.
        learn_rate
            Indicates how much the bias and weight should adjust.
        """        
        for i in range(len(self.output)):
            self.bias[i] += self.error[i] * learn_rate
            for j in range(self.num_weights):
                self.weights[i][j] += self.error[i] * learn_rate * input[j]
        return
    
    def print(self, precision):
        from time import sleep
        """print: 
            Prints the dense_layers values, such as output, error, bias, weights and the
            number of nodes and weights per node. sleep is used to avoid faulty prints.

        Parameters
        ----------
        precision
            number of decimals.
        """
        self.round(precision)
        print("--------------------------------------------------------")
        print(f"Number of nodes:\t\t{self.num_nodes}")
        print(f"Number of weights per node:\t{self.num_weights}")
        sleep(0.5)
        print(f"Output:\t\t\t\t{self.output}")
        print(f"Error:\t\t\t\t{self.error}")
        sleep(0.5)
        print(f"Bias:\t\t\t\t{self.bias}")
        print(f"Weights:")
        for i in range(self.num_nodes):
            print(f"\t\t\tNode {i+1}:\t{self.weights[i]}")
            sleep(0.2)
        print("--------------------------------------------------------")
        sleep(0.5)

    def round(self, precision):
        """round:
            Rounds all the values to number of decimals.

        Parameters
        ----------
        precision
            number of decimals.
        """
        self.bias = [round(x, precision) for x in self.bias]
        self.weights = [[round(x, precision) for x in sublist] for sublist in self.weights]
        self.output = [round(x, precision) for x in self.output]
        self.error = [round(x, precision) for x in self.error]

if __name__ == "__main__":
    print("Executing module dense_layer.py, not recommended.")