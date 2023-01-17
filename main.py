from ann import Ann
"""
TODO: Add ANN class with, optimize, train.
TODO: method to resize number of nodes in a given layer
TODO: Safety checks, incase no-layers less or equal to 1
"""

def read_file(filename):
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
        

def main():
    train_in, train_out = read_file("train_data")
    

    #d1 = Dense_layer(5, 3) #number of outputs, number of inputs
    ann = Ann(4,1,3,6)
    from time import sleep
    #ann.feedforward(train_in)
    #ann.backpropagate(train_out)
    ann.set_training_data(train_in, train_out)
    
    ann.train(train_in, train_out, 150000, 0.1)
    print("training done!")
    sleep(2)
    """    #for _ in range(10000):
    ann.feedforward([1,1,1,0])
    ann.backpropagate([1])
    ann.output_layer_.print(2)
    for i in range(len(ann.hidden_layer_)-1, -1, -1):
        print(f"|\tLAYER: {i}\t|")
        
        ann.hidden_layer_[i].print(2)
        sleep(1)
    
    print("\nFINAL STEP\n")
    print("Should be 1")
    print(ann.output_layer_.output[0])"""
    for _ in range(len(train_in)):
        ann.feedforward(train_in[_])
        ann.backpropagate(train_out[_])
        print(f"Input:\t\t{train_in[_]}")
        print(f"Reference:\t{train_out[_]}")
        print(f"Prediction:\t{ann.output_layer_.output[0]}, Error:{ann.output_layer_.error[0]}")
        print("----------------------------------")

if __name__ == "__main__":
    main()