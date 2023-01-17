from dense_layer import Dense_layer

class Ann:
    def __init__(self, num_inputs, num_outputs, num_hidden_layers, num_hidden_nodes):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden_nodes = num_hidden_nodes
        self.num_hidden_layers = num_hidden_layers
        
        self.hidden_layer_ = []
        self.output_layer_ = Dense_layer
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
            for layer in range(1, self.num_hidden_layers): # maybe an error, took layer 0 twice
                self.hidden_layer_[layer].feedforward(self.hidden_layer_[layer-1].output)
        self.output_layer_.feedforward(self.hidden_layer_[-1].output)

    def backpropagate(self, reference):
        self.output_layer_.backprop_outer(reference)
        self.hidden_layer_[-1].backprop_hidden(self.output_layer_)
        if self.num_hidden_layers > 1:
            for layer in range(self.num_hidden_layers-1, 0, -1):
                self.hidden_layer_[layer-1].backprop_hidden(self.hidden_layer_[layer])

    def optimize(self, input, learn_rate):
        self.hidden_layer_[0].optimize(input, learn_rate)
        if self.num_hidden_layers > 1:
            for layer in range(1, self.num_hidden_layers): # maybe an error, took layer 0 twice
                self.hidden_layer_[layer].optimize(self.hidden_layer_[layer-1].output, learn_rate)
                
        self.output_layer_.optimize(self.hidden_layer_[-1].output, learn_rate)

    def train(self, input, reference, epochs=1000, learn_rate=0.1):
        for _ in range(epochs):
            #shuffle training data, unzip and feedforward, backprop, optimize repeat...
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
        pass
        

    """ def set_traning_data(self, train_in, train_out):
        self.train_in_ = train_in 
        self.train_out_ = train_out"""

if __name__ == "__main__":
    print("Not the main file")