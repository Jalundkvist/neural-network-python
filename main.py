from ann import Ann
from gpio import GPIO
from time import sleep

"""
TODO: Accuracy sometimes above 99%, sometimes 60%
TODO: Clean up
TODO: Update comments
TODO: Add RPI buttons & LED.
TODO: Train: Try except för att köra programmet igen.
"""

EPOCHS = 10000
LEARN_RATE = 0.0255
DATA_FILE = "train_data"
OUTPUT_FILE = "output"

def main():
    ann = Ann(4,1,3,4)
    train(ann)
    # create an instance for each button
    button1 = GPIO(17, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    button2 = GPIO(27, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    button3 = GPIO(22, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    button4 = GPIO(23, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    buttons = [button1, button2, button3, button4]
    # create an instance for the led
    led = GPIO(24, GPIO.OUT)
    while(True):
        button_values = [button.input() for button in buttons]
        led.output(ann.predict(button_values)[0])
        sleep(5)

def train(NeuralNetwork):
    import time
    train_in, train_out = NeuralNetwork.read_file(DATA_FILE)
    NeuralNetwork.set_training_data(train_in, train_out)
    print("\033c", end="")
    print("---------------------------------")
    print("Training initiating!\n\n")
    print(f"Number of epochs:\t{EPOCHS}")
    print(f"Learnrate:\t\t{LEARN_RATE}")
    start_time = time.time()
    NeuralNetwork.train(train_in, train_out, EPOCHS, LEARN_RATE)
    end_time = time.time()
    print(f"\nTraining done!")
    print(f"Complete training time was: {round(end_time - start_time, 1)}s")
    print("---------------------------------")


if __name__ == "__main__":
    main()