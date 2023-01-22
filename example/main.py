import sys
sys.path.append("..\src")
from ann import Ann
from gpio import GPIO
from time import sleep

EPOCHS = 10000
LEARN_RATE = 0.0255
ACCURACY = 98 # Percentage
DATA_FILE = "train_data"
OUTPUT_FILE = "output"

"""
This code creates an instance of the Ann class, which is used to create a neural network.
It then calls the local train() function to train the neural network using data from the file "train_data". 
The number of epochs and learning rate are set to 10000 and 0.0255 respectively. 

After training, four instances of the GPIO class are created for each button, as well as one instance for the LED. 
The program then enters a loop where it reads in values from each button and outputs a prediction from the neural network to the LED.
This loop runs indefinitely until interrupted by an external force. 

The train() function takes in a NeuralNetwork parameter and reads in data from a file specified by DATA_FILE.
It then sets this data as training data for the neural network before printing out information about the training process, 
such as number of epochs and learning rate, before beginning training. 
Once training is complete, it prints out how long it took to complete training. 

For this example the data for a 4 to 1 XOR gate has been used. Acceptable results of 99.x% accuracy with
10k epochs and learn rate set to 0.0255, 3 hidden layers with 5-4-6 nodes respectively by 
using the "resize_hidden_layer" method.
If the training fails you will be prompted to repeat training (might occur depending on initial weights & biases).
"""
def main():
    ann = Ann(4,1,3,4)
    train(ann)
    # create an instance for each button and adds them to a list.
    button1 = GPIO(17, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    button2 = GPIO(27, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    button3 = GPIO(22, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    button4 = GPIO(23, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    buttons = [button1, button2, button3, button4]
    # create an instance for the led
    led = GPIO(24, GPIO.OUT)
    
    while(True):
        button_values = [button.input() for button in buttons]
        #predict returns a list, since only 1 output, index 0.
        led.output(ann.predict(button_values)[0])
        sleep(0.5)

def train(NeuralNetwork):
    """train:
    Local function to train the neural network.
    Prints time it took to train.

    Parameters
    ----------
    NeuralNetwork
        The desired network to be trained.
    """    
    import time
    train_in, train_out = NeuralNetwork.read_file(DATA_FILE)
    NeuralNetwork.set_training_data(train_in, train_out)
    print("\033c", end="")
    print("---------------------------------")
    print("Training initiating!\n")
    print("Executing resize_hidden_layer\n")
    # resizes first layer to 5
    NeuralNetwork.resize_hidden_layer(0,5)
    NeuralNetwork.resize_hidden_layer(2,6)
    print(f"Number of epochs:\t{EPOCHS}")
    print(f"Learnrate:\t\t{LEARN_RATE}")
    start_time = time.time()
    NeuralNetwork.train(EPOCHS, LEARN_RATE, ACCURACY)
    end_time = time.time()
    print(f"Training done!")
    print(f"Complete training time was: {round(end_time - start_time, 1)}s\n")
    print(f"Check {OUTPUT_FILE}.txt for full report")
    print("---------------------------------\n\n")
    sleep(5)

if __name__ == "__main__":
    main()