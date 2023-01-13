from dense_layer import Dense_layer

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
            train_out.append(int(input_output[1]))
    except FileNotFoundError:
        print("The specified file could not be found.")
        return -1
    except:
        print("An error occurred while trying to read the file.")
        return -1
    else:
        return train_in, train_out
        

def main():  # Testa

    train_in, train_out = read_file("train_data")
    d1 = Dense_layer(5, 4)
    train_in = [1, 1, 1, 1]
    train_out = [2, 2, 2, 2, 2]
    for _ in range(100):
        d1.feedforward(train_in)
        d1.backprop_outer(train_out)
        d1.optimize(train_in, 3)
    print()
    print("Welcome to this neural network training program!\n")
    # d1.print_vals()
    print()
    d1.print(precision=4)


if __name__ == "__main__":
    main()