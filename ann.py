from dense_layer import Dense_layer

class ann:
    def __init__(self, num_hidden_layers, num_hidden_nodes):
        self.hidden_layer_ = []
        self.output_layer_ = Dense_layer()
        for _ in range(num_hidden_layers):
            self.hidden_layer_.append(Dense_layer(num_hidden_nodes)) 

    def set_traning_data(self, train_in, train_out):
        self.train_in_ = train_in 
        self.train_out_ = train_out