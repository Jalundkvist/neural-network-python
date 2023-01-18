from dense_layer import Dense_layer

class Ann:
    def __init__(self, num_inputs, num_outputs, num_hidden_layers, num_hidden_nodes):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden_nodes = num_hidden_nodes
        self.num_hidden_layers = num_hidden_layers
        self.precision = 0
        
        #self.hidden_layer_ = []
        #self.output_layer_ = Dense_layer
        self.set_layers()
        
        self.training_data = []
        self.reference_data = []

    def set_layers(self):
        self.hidden_layer_ = []
        self.hidden_layer_.append(Dense_layer(self.num_hidden_nodes, self.num_inputs))
        for _ in range(self.num_hidden_layers-1):
            self.hidden_layer_.append(Dense_layer(self.num_hidden_nodes, self.num_hidden_nodes))
        self.output_layer_ = Dense_layer(self.num_outputs, self.num_hidden_nodes)
        
    def set_training_data(self, training_data, reference_data):
        if len(training_data) == len(reference_data):
            self.training_data = training_data
            self.reference_data = reference_data
        elif len(training_data) < len(reference_data):
            self.training_data = training_data
            self.reference_data = reference_data[0:len(training_data)]
        else:
            self.reference_data = reference_data
            self.training_data = training_data[0:len(reference_data)]

    def feedforward(self, input):
        self.hidden_layer_[0].feedforward(input)
        if self.num_hidden_layers > 1:
            for layer in range(1, self.num_hidden_layers):
                self.hidden_layer_[layer].feedforward(self.hidden_layer_[layer-1].output)
        self.output_layer_.feedforward(self.hidden_layer_[-1].output)

    def backpropagate(self, reference):
        self.output_layer_.backprop_outer(reference)
        self.hidden_layer_[-1].backprop_hidden(self.output_layer_)
        if self.num_hidden_layers > 1:
            for layer in range(len(self.hidden_layer_)-1, 0, -1):
                self.hidden_layer_[layer-1].backprop_hidden(self.hidden_layer_[layer])

    def optimize(self, input, learn_rate):
        self.hidden_layer_[0].optimize(input, learn_rate)
        if self.num_hidden_layers > 1:
            for layer in range(1, self.num_hidden_layers):
                self.hidden_layer_[layer].optimize(self.hidden_layer_[layer-1].output, learn_rate)
                
        self.output_layer_.optimize(self.hidden_layer_[-1].output, learn_rate)

    def train(self, input, reference, epochs=1000, learn_rate=0.1):
        for _ in range(epochs):
            training_order = self.shuffle()
            for index in training_order:
                self.feedforward(input[index])
                self.backpropagate(reference[index])
                self.optimize(input[index], learn_rate)

    def shuffle(self):
        from random import shuffle
        indices = list(range(len(self.training_data)))
        shuffle(indices)
        return indices
    
    def resize_hidden_layer(self, layer, number_of_nodes):
        try:
            # if layer is inbetween
            if layer > 0 and self.hidden_layer_[layer] != self.hidden_layer_[-1]:
                self.hidden_layer_[layer].resize(number_of_nodes, self.hidden_layer_[layer-1].num_nodes)
                self.hidden_layer_[layer+1].resize(self.hidden_layer_[layer+1].num_nodes, self.hidden_layer_[layer].num_nodes)
            # if first hidden layer
            elif layer == 0:
                self.hidden_layer_[layer].resize(number_of_nodes, self.num_inputs)
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

    def within_range(self, prediction,reference):
        if abs(prediction - reference) <= 0.1:
            return round(prediction)
        else:
            return prediction
    
    def read_file(self, filename="train_data"):
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
        
    def write_file(self, epochs, learn_rate, filename="output"):
        total_error = 0
        total_predictions = len(self.training_data)
        with open(f"{filename}.txt", "w") as file:
            file.write("OUTPUT AFTER TRAINING\n\n")
            file.write(f"Number of inputs:\t{self.num_inputs}\n")
            file.write(f"Number of outputs:\t{self.num_outputs}\n")
            file.write(f"Number of hidden layers: {self.num_hidden_layers}\n")
            for i in range(self.num_hidden_layers):
                file.write(f"Hidden layer {i}: {self.hidden_layer_[i].num_nodes} nodes\n")
            
            file.write(f"Epochs:\t{epochs}\n")
            file.write(f"Learn rate:\t{learn_rate}\n\n")
            self.output_layer_.round(2)
            file.write(f"Final output weight: {self.output_layer_.weights}\n")
            file.write(f"Final output bias:\t {self.output_layer_.bias}\n")
            
            file.write("------------------------------------------\n")
            for _ in range(len(self.training_data)):
                self.feedforward(self.training_data[_])
                self.backpropagate(self.reference_data[_])
                self.output_layer_.round(3)
                file.write(f"Input:\t\t{self.training_data[_]}\n")
                file.write(f"Reference:\t{self.reference_data[_]}\n")
                checked_val = []
                for i in range(len(self.output_layer_.output)):
                    checked_val = self.within_range(self.output_layer_.output[i], self.reference_data[_][i])
                    file.write(f"Prediction:\t{checked_val}\n")
                    file.write(f"Error:\t{self.output_layer_.error[i]:.3f}\n")
                    total_error += abs(self.output_layer_.error[i])
                    
                file.write("------------------------------------------\n")
            self.precision = round((100 - (total_error/total_predictions)*100),2)
            file.write(f"Accuracy:\t{self.precision}%")

if __name__ == "__main__":
    print("Not the main file")