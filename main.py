# Skulle du kunn
from dense_layer import Dense_layer

def main():# Testa
    d1 = Dense_layer(5,4)
    train_in = [1,1,1,1]
    train_out = [2,2,2,2,2] 
    for i in range(100):
        d1.feedforward(train_in)
        d1.backprop_outer(train_out)
        d1.optimize(train_in, 0.7)  #0.5 = 0 error 0 output 
    print()
    print("Wällkom to this näral netwörk!\n")
    #d1.print_vals()
    print()
    d1.print(precision=4)

if __name__ == "__main__":
    main()
    
    
    """class ann:

    def __init__(self, num_layers, num_hidden_nodes):
        self.hidden_layer_ = []
        self.output_layer_ = dense_layer()
        for layer in range(num_layers): #lalalayer
            self.hidden_layer_.append(dense_layer(num_hidden_nodes)) 
    
    def set_traning_data(self, num_input, num_output):
        self.train_in_ = num_input 
        self.train_out_ = num_output
"""