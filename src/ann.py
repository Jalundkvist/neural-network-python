from dense_layer import Dense_layer

class Ann:
    """
    Artificial Neural Network (Ann) class that uses a feedforward-backpropagation algorithm 
    to train a neural network model using Dense_layer objects.
    
    Parameters
    ----------
    num_inputs:
        Number of inputs for the neural network.
    num_outputs:
        Number of outputs for the neural network.
    num_hidden_layers:
        Number of hidden layers in the neural network.
    num_hidden_nodes:
        Number of nodes in each hidden layer.
    """   
    def __init__(self, num_inputs, num_outputs, num_hidden_layers, num_hidden_nodes):
        assert num_inputs > 0, "Number of inputs must be greater than 0"
        assert num_outputs > 0, "Number of outputs must be greater than 0"
        assert num_hidden_layers > 0, "Number of hidden layers must be greater than 0"
        assert all(val > 0 for val in num_hidden_nodes), "Number of hidden nodes must be greater than 0"

        self._num_inputs = num_inputs
        self._num_outputs = num_outputs
        self._num_hidden_layers = num_hidden_layers
        self._accuracy = 0
        self._training_complete = False
        self._set_layers(num_hidden_nodes)
        
        self._input_data = []
        self._reference_data = []

    def _set_layers(self, num_hidden_nodes):
        """set_layers: 
            Constructs the neural network by adding Dense_layer objects to lists.
        
        Parameters
        ----------
        num_hidden_nodes
            number of nodes in all the hidden layers.
        """        
        self.hidden_layer_ = []
        self.hidden_layer_.append(Dense_layer(num_hidden_nodes, self._num_inputs))
        if self._num_hidden_layers > 1:
            for _ in range(self._num_hidden_layers-1):
                self.hidden_layer_.append(Dense_layer(num_hidden_nodes, num_hidden_nodes))
        self.output_layer_ = Dense_layer(self._num_outputs, num_hidden_nodes)
        return
        
    def set_training_data(self, input_data, reference_data):
        """set_training_data:
            Adds the training data to the neural network.

        Parameters
        ----------
        input_data
            A list containing lists of input data.
        reference_data
            A list containing lists of reference data.
        """        
        if len(input_data) == len(reference_data):
            self._input_data = input_data
            self._reference_data = reference_data
        elif len(input_data) < len(reference_data):
            self._input_data = input_data
            self._reference_data = reference_data[0:len(input_data)]
        else:
            self._reference_data = reference_data
            self._input_data = input_data[0:len(reference_data)]
        return

    def feedforward(self, input_data):
        """feedforward:
            Calculates the output result by feeding input data through
            all the layers in the network.

        Parameters
        ----------
        input_data
            List of the input training values.
        """        
        self.hidden_layer_[0].feedforward(input_data)
        if self._num_hidden_layers > 1:
            for layer in range(1, self._num_hidden_layers):
                self.hidden_layer_[layer].feedforward(self.hidden_layer_[layer-1].output)
        self.output_layer_.feedforward(self.hidden_layer_[-1].output)

    def backpropagate(self, reference_data):
        """backpropagate:
            Calculates the error for all the nodes from output layer to 
            first dense layer.

        Parameters
        ----------
        reference_data
            List of the output training values.
        """        
        self.output_layer_.backprop_outer(reference_data)
        self.hidden_layer_[-1].backprop_hidden(self.output_layer_)
        if self._num_hidden_layers > 1:
            for layer in range(len(self.hidden_layer_)-1, 0, -1):
                self.hidden_layer_[layer-1].backprop_hidden(self.hidden_layer_[layer])

    def optimize(self, input_data, learn_rate):
        """optimize:
            Optimizes the weights and biases in the neural network model 
            from first dense layer to output layer.

        Parameters
        ----------
        input_data
            List of the input training values.
        learn_rate
            Which rate the model learn, too high = overfitting, too low = underfitting.
        """        
        self.hidden_layer_[0].optimize(input_data, learn_rate)
        if self._num_hidden_layers > 1:
            for layer in range(1, self._num_hidden_layers):
                self.hidden_layer_[layer].optimize(self.hidden_layer_[layer-1].output, learn_rate)
                
        self.output_layer_.optimize(self.hidden_layer_[-1].output, learn_rate)

    def train(self, epochs=1000, learn_rate=0.1, accuracy_threshold=98, output_file="output"):
        """train: 
            Trains the neural network model.

        Parameters
        ----------
        epochs
            Number of training cycles.
        learn_rate
            Which rate the model learn, too high = overfitting, too low = underfitting.
        accuracy_threshold
            The users desired accuracy of the network.
        output_file
            The file that the output data gets written to.
        """
        if len(self._input_data) == 0:
            print("Please make sure you've set the training data with\
                the set_training_data method")
            return -1
        else:
            for _ in range(epochs):
                training_order = self.shuffle()
                for i in training_order:
                    self.feedforward(self._input_data[i])
                    self.backpropagate(self._reference_data[i])
                    self.optimize(self._input_data[i], learn_rate)
            self._write_file(epochs, learn_rate, output_file)
                    
            if self._accuracy > accuracy_threshold:
                print(f"Accuracy: {self._accuracy}%\n")
                self._training_complete = True
                return
                
            while self._accuracy < accuracy_threshold:
                print(f"\nAccuracy is: {self._accuracy}% and is")
                print(f"below the threshhold of {accuracy_threshold}%\n")
                answer = input("Do you wish to repeat the training? y/n \n")
                # Checks if input is str
                if isinstance(answer, str):
                    if answer.lower().startswith("y"):
                        self.train(epochs, learn_rate, accuracy_threshold, output_file)
                    elif answer.lower().startswith("n"):
                        quit()
                    else:
                        print("Invalid input, please enter 'y' or 'n'\n")
                else:
                    print("Invalid input, please enter 'y' or 'n'\n")
                

    def shuffle(self):
        """shuffle:
            Returns a list of random index numbers based
            on number of input data for optimal training.
            
            k equals the size of the list returned
        """
        from random import choices
        return (choices(range(len(self._input_data)), k=len(self._input_data)))
    
    def resize_hidden_layer(self, layer, number_of_nodes):
        """resize_hidden_layer:
             Resizes the specified layer (index based)
             with the number of nodes

        Parameters
        ----------
        layer
            One of the avaible hidden layers (counts from 0).
        number_of_nodes
            Number of nodes to update a given hidden layer.

        Raises
        ------
        IndexError
            If the specified layer is not in range of index.
        """        
        try:
            if number_of_nodes <= 0:
                print("Number of nodes can not be below 0, nodes set to 1.")
                number_of_nodes = 1
            # if only one layer (first and last)
            if  layer == 0 and self._num_hidden_layers == 1:
                self.hidden_layer_[layer].resize(number_of_nodes, self._num_inputs)
                self.output_layer_.resize(self.output_layer_.num_nodes, self.hidden_layer_[layer].num_nodes)
            # if first hidden layer
            elif layer == 0 and self.hidden_layer_[layer] != self.hidden_layer_[-1]:
                self.hidden_layer_[layer].resize(number_of_nodes, self._num_inputs)
                self.hidden_layer_[layer+1].resize(self.hidden_layer_[layer+1].num_nodes, self.hidden_layer_[layer].num_nodes)
            # if layer is in between
            elif layer > 0 and self.hidden_layer_[layer] != self.hidden_layer_[-1]:
                self.hidden_layer_[layer].resize(number_of_nodes, self.hidden_layer_[layer-1].num_nodes)
                self.hidden_layer_[layer+1].resize(self.hidden_layer_[layer+1].num_nodes, self.hidden_layer_[layer].num_nodes)
            # if last hidden layer
            elif self.hidden_layer_[layer] == self.hidden_layer_[-1]:
                self.hidden_layer_[layer].resize(number_of_nodes, self.hidden_layer_[layer-1].num_nodes)
                self.output_layer_.resize(self.output_layer_.num_nodes, self.hidden_layer_[layer].num_nodes)
                
        except IndexError as err:
            print("\n")
            raise IndexError(f"An error occurred: {err} in resize_hidden_layer method\n\n")
        except Exception:
            print("Uknown exception occured in resize_hidden_layer method.\n\n")

    def read_file(self, filename="train_data"):
        """read_file: 
            Reads the input data.

        Parameters
        ----------
        filename, optional
            name of file containing training data,
            by default "train_data".

        Returns
        -------
            Returns the read training data,
        """        
        try:
            with open(f"{filename}.txt", "r") as file:
                data = file.read()
                lines = data.split("\n")
            
            train_in = []
            train_out = []

            for line in lines:
                input_output = line.split(",")
                train_in.append(list(map(int, input_output[0].split())))
                train_out.append(list(map(int, input_output[1])))
        except FileNotFoundError:
            print("The specified file could not be found.")
            return -1
        except:
            print("An error occurred while trying to read the file.")
            return -1
        else:
            return train_in, train_out
        
    def _write_file(self, epochs, learn_rate, filename):
        """_write_file:
            Writes the results to a text file

        Parameters
        ----------
        epochs
            Number of training cycles.
        learn_rate
            Which rate the model learn, too high = overfitting, too low = underfitting.
        filename, optional
            filename where data is written to, by default "output"
        """        
        total_error = 0
        total_predictions = len(self._input_data)
        with open(f"{filename}.txt", "w") as file:
            file.write("OUTPUT AFTER TRAINING\n\n")
            file.write(f"Number of inputs:\t{self._num_inputs}\n")
            file.write(f"Number of outputs:\t{self._num_outputs}\n")
            file.write(f"Number of hidden layers: {self._num_hidden_layers}\n")
            for i in range(self._num_hidden_layers):
                file.write(f"Hidden layer {i}: {self.hidden_layer_[i].num_nodes} nodes\n")

            file.write(f"Epochs:\t{epochs}\n")
            file.write(f"Learn rate:\t{learn_rate}\n\n")
            self.output_layer_.round(2)
            file.write(f"Final output weight: {self.output_layer_.weights}\n")
            file.write(f"Final output bias:\t {self.output_layer_.bias}\n")

            file.write("------------------------------------------\n")
            for _ in range(len(self._input_data)):
                self.feedforward(self._input_data[_])
                self.backpropagate(self._reference_data[_])
                self.output_layer_.round(3)
                file.write(f"input_data:\t\t{self._input_data[_]}\n")
                file.write(f"reference_data:\t{self._reference_data[_]}\n")

                for i in range(len(self.output_layer_.output)):
                    file.write(f"Prediction:\t{round(self.output_layer_.output[i])}\n")
                    file.write(f"Error:\t{self.output_layer_.error[i]:.3f}\n")
                    total_error += abs(self.output_layer_.error[i])
                file.write("------------------------------------------\n")
            self._accuracy = round((100 - (total_error/total_predictions)*100),2)
            file.write(f"Accuracy:\t{self._accuracy}%")

    def predict(self, input):
        """predict:
            Predicts the output based on the input.

        Parameters
        ----------
        input
            list of values
            based on number of inputs.

        Returns
        -------
            returns a list of predicted values
            based on number of outputs.
        """        
        if self._training_complete:
            self.feedforward(input)
            result = []
            for i in range(self._num_outputs):
                result.append(round(self.output_layer_.output[i]))
            print(f"Input: {input}")
            print(f"Predicted value: {result}")
            return result
        else:
            print("Please train the neural network to acceptable accuracy before using predict")
            return -1
        
if __name__ == "__main__":
    print("Executing module ann.py, not recommended.")