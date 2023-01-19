from dense_layer import Dense_layer

class Ann:
    def __init__(self, num_inputs, num_outputs, num_hidden_layers, num_hidden_nodes):

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden_nodes = num_hidden_nodes
        self.num_hidden_layers = num_hidden_layers
        self.accuracy = 100
        self.set_layers()
        
        self.input_data = []
        self.reference_data_data = []

    def set_layers(self):
        """set_layers _summary_
        """        
        self.hidden_layer_ = []
        self.hidden_layer_.append(Dense_layer(self.num_hidden_nodes, self.num_inputs))
        # if num layers > 1
        for _ in range(self.num_hidden_layers-1):
            self.hidden_layer_.append(Dense_layer(self.num_hidden_nodes, self.num_hidden_nodes))
        self.output_layer_ = Dense_layer(self.num_outputs, self.num_hidden_nodes)
        
    def set_training_data(self, input_data, reference_data_data):
        """set_training_data _summary_

        Parameters
        ----------
        input_data
            _description_
        reference_data_data
            _description_
        """        
        if len(input_data) == len(reference_data_data):
            self.input_data = input_data
            self.reference_data_data = reference_data_data
        elif len(input_data) < len(reference_data_data):
            self.input_data = input_data
            self.reference_data_data = reference_data_data[0:len(input_data)]
        else:
            self.reference_data_data = reference_data_data
            self.input_data = input_data[0:len(reference_data_data)]

    def feedforward(self, input_data):
        """feedforward _summary_

        Parameters
        ----------
        input_data
            _description_
        """        
        self.hidden_layer_[0].feedforward(input_data)
        if self.num_hidden_layers > 1:
            for layer in range(1, self.num_hidden_layers):
                self.hidden_layer_[layer].feedforward(self.hidden_layer_[layer-1].output)
        self.output_layer_.feedforward(self.hidden_layer_[-1].output)

    def backpropagate(self, reference_data):
        """backpropagate _summary_

        Parameters
        ----------
        reference_data
            _description_
        """        
        self.output_layer_.backprop_outer(reference_data)
        self.hidden_layer_[-1].backprop_hidden(self.output_layer_)
        if self.num_hidden_layers > 1:
            for layer in range(len(self.hidden_layer_)-1, 0, -1):
                self.hidden_layer_[layer-1].backprop_hidden(self.hidden_layer_[layer])

    def optimize(self, input_data, learn_rate):
        """optimize _summary_

        Parameters
        ----------
        input_data
            _description_
        learn_rate
            _description_
        """        
        self.hidden_layer_[0].optimize(input_data, learn_rate)
        if self.num_hidden_layers > 1:
            for layer in range(1, self.num_hidden_layers):
                self.hidden_layer_[layer].optimize(self.hidden_layer_[layer-1].output, learn_rate)
                
        self.output_layer_.optimize(self.hidden_layer_[-1].output, learn_rate)

    def train(self, input_data, reference_data, epochs=1000, learn_rate=0.1, output_file ="output"):
        """train _summary_

        Parameters
        ----------
        input_data
            _description_
        reference_data
            _description_
        epochs, optional
            _description_, by default 1000
        learn_rate, optional
            _description_, by default 0.1
        """        
        for _ in range(epochs):
            training_order = self.shuffle()
            for index in training_order:
                self.feedforward(input_data[index])
                self.backpropagate(reference_data[index])
                self.optimize(input_data[index], learn_rate)
        self.write_file(epochs, learn_rate, output_file)
        if self.accuracy > 98:
            print(f"Accuracy: {self.accuracy}%\n")
        while self.accuracy < 98:
            print(f"Accuracy is: {self.accuracy}% and is below threshhold of 98%\n")
            answer = input("Do you wish to repeat the training or quit? y/n \n")
            if isinstance(answer, str):
                if answer.lower().startswith("y"):
                    self.train(input_data, reference_data, epochs, learn_rate)
                    break
                elif answer.lower().startswith("n"):
                    quit()
                else:
                    print("Invalid input, please enter 'y' or 'n'\n")

            else:
                print("Invalid input, please enter 'y' or 'n'\n")
                

    def shuffle(self):
        """shuffle _summary_

        Returns
        -------
            _description_
        """
        import random
        return (random.choices(range(len(self.input_data)), k=len(self.input_data)))
    
    def resize_hidden_layer(self, layer, number_of_nodes):
        """resize_hidden_layer _summary_

        Parameters
        ----------
        layer
            _description_
        number_of_nodes
            _description_

        Raises
        ------
        IndexError
            _description_
        """        
        try:
            # if only one layer (first and last)
            if  layer == 0 and self.num_hidden_layers == 1:
                self.hidden_layer_[layer].resize(number_of_nodes, self.num_inputs)
                self.output_layer_.resize(self.output_layer_.num_nodes, self.hidden_layer_[layer].num_nodes)
            # if first hidden layer
            elif layer == 0 and self.hidden_layer_[layer] != self.hidden_layer_[-1]:
                self.hidden_layer_[layer].resize(number_of_nodes, self.num_inputs)
                self.hidden_layer_[layer+1].resize(self.hidden_layer_[layer+1].num_nodes, self.hidden_layer_[layer].num_nodes)
            # if layer is inbetween
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

    def within_range(self, prediction,reference_data):
        """within_range _summary_

        Parameters
        ----------
        prediction
            _description_
        reference_data
            _description_

        Returns
        -------
            _description_
        """        
        if abs(prediction - reference_data) <= 0.1:
            return round(prediction)
        else:
            return prediction
    
    def read_file(self, filename="train_data"):
        """read_file takes input_data from the inpur file

        Parameters
        ----------
        filename, optional
            _description_, by default "train_data"

        Returns
        -------
            _description_
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
        
    def write_file(self, epochs, learn_rate, filename):
        """write_file _summary_

        Parameters
        ----------
        epochs
            _description_
        learn_rate
            _description_
        filename, optional
            _description_, by default "output"
        """        
        total_error = 0
        total_predictions = len(self.input_data)
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
            for _ in range(len(self.input_data)):
                self.feedforward(self.input_data[_])
                self.backpropagate(self.reference_data_data[_])
                self.output_layer_.round(3)
                file.write(f"input_data:\t\t{self.input_data[_]}\n")
                file.write(f"reference_data:\t{self.reference_data_data[_]}\n")
                checked_val = []
                for i in range(len(self.output_layer_.output)):
                    checked_val = self.within_range(self.output_layer_.output[i], self.reference_data_data[_][i])
                    file.write(f"Prediction:\t{checked_val}\n")
                    file.write(f"Error:\t{self.output_layer_.error[i]:.3f}\n")
                    total_error += abs(self.output_layer_.error[i])
                file.write("------------------------------------------\n")
            self.accuracy = round((100 - (total_error/total_predictions)*100),2)
            file.write(f"Accuracy:\t{self.accuracy}%")

    def predict(self, input):
        self.feedforward(input)
        result = []
        for i in range(self.num_outputs):
            result.append(round(self.output_layer_.output[i]))
        print(f"Input: {input}")
        print(f"Predicted value: {result}")
        return result
        
        

if __name__ == "__main__":
    print("Not the main file")